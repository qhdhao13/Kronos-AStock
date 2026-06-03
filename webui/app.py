import os
import re
import time
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: 无法导入 Kronos 模型，将使用模拟数据演示")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': '轻量模型，预测速度快'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': '小型模型，速度与效果均衡（推荐）'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': '基础模型，预测质量更高但较慢'
    }
}

# K 线周期与复权方式映射（东方财富 API 参数）
KLINE_PERIOD_MAP = {
    'daily': ('101', 'daily', '日K'),
    '5min': ('5', '5min', '5分钟'),
}
ADJUST_MAP = {
    'qfq': ('1', 'qfq', '前复权'),
    'hfq': ('2', 'hfq', '后复权'),
    'none': ('0', 'none', '不复权'),
}


def get_data_dir():
    """返回并确保 data 目录存在"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def normalize_stock_symbol(symbol):
    """校验并规范化 A 股 6 位代码"""
    symbol = str(symbol).strip()
    if not re.fullmatch(r'\d{6}', symbol):
        raise ValueError('股票代码必须是 6 位数字，例如 000338、600519')
    return symbol


def symbol_to_secid(symbol):
    """将 A 股代码转为东方财富 secid（沪 1. / 深 0.）"""
    symbol = normalize_stock_symbol(symbol)
    if symbol.startswith(('5', '6', '9')):
        return f'1.{symbol}'
    return f'0.{symbol}'


def _parse_eastmoney_klines(klines):
    """解析东方财富 K 线字符串列表"""
    rows = []
    for line in klines:
        parts = line.split(',')
        rows.append({
            'timestamps': parts[0],
            'open': float(parts[1]),
            'close': float(parts[2]),
            'high': float(parts[3]),
            'low': float(parts[4]),
            'volume': float(parts[5]),
            'amount': float(parts[6]),
        })
    df = pd.DataFrame(rows)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    return df.sort_values('timestamps').reset_index(drop=True)


def _fetch_from_eastmoney(symbol, klt, fqt):
    """尝试多个东方财富接口（HTTPS 常被断开，HTTP 镜像更稳定）"""
    secid = symbol_to_secid(symbol)
    params = {
        'secid': secid,
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': klt,
        'fqt': fqt,
        'end': '20500101',
        'lmt': '10000',
    }
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ),
        'Referer': 'https://quote.eastmoney.com/',
        'Accept': 'application/json, text/plain, */*',
    }
    urls = [
        'http://push2his.eastmoney.com/api/qt/stock/kline/get',
        'https://push2his.eastmoney.com/api/qt/stock/kline/get',
        'http://63.push2his.eastmoney.com/api/qt/stock/kline/get',
    ]

    errors = []
    session = requests.Session()
    for url in urls:
        for attempt in range(2):
            try:
                resp = session.get(url, params=params, headers=headers, timeout=25)
                resp.raise_for_status()
                klines = resp.json().get('data', {}).get('klines') or []
                if not klines:
                    raise ValueError('返回空数据')
                return _parse_eastmoney_klines(klines)
            except Exception as e:
                errors.append(f'{url}: {e}')
                time.sleep(1)
    raise ConnectionError('东方财富接口均失败；' + ' | '.join(errors[-3:]))


def _fetch_from_akshare(symbol, period, adjust):
    """AKShare 备用数据源（可选）"""
    try:
        import akshare as ak
    except ImportError:
        raise ConnectionError('AKShare 未安装')

    ak_adjust = {'qfq': 'qfq', 'hfq': 'hfq', 'none': ''}[adjust]
    ak_period = {'daily': 'daily', '5min': '5'}[period]

    for attempt in range(2):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period=ak_period, adjust=ak_adjust)
            if df is None or df.empty:
                raise ValueError('返回空数据')
            df = df.rename(columns={
                '日期': 'timestamps',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
            })
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            return df.sort_values('timestamps').reset_index(drop=True)
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise ConnectionError(f'AKShare 失败: {last}')


def fetch_a_share_kline(symbol, period='daily', adjust='qfq'):
    """拉取 A 股 K 线，自动切换数据源"""
    if period not in KLINE_PERIOD_MAP:
        raise ValueError(f'不支持的周期: {period}')
    if adjust not in ADJUST_MAP:
        raise ValueError(f'不支持的复权方式: {adjust}')

    symbol = normalize_stock_symbol(symbol)
    klt, _, _ = KLINE_PERIOD_MAP[period]
    fqt, _, _ = ADJUST_MAP[adjust]

    errors = []
    try:
        return _fetch_from_eastmoney(symbol, klt, fqt)
    except Exception as e:
        errors.append(str(e))

    try:
        return _fetch_from_akshare(symbol, period, adjust)
    except Exception as e:
        errors.append(str(e))

    raise ConnectionError('所有数据源均失败。' + '；'.join(errors))


def save_stock_csv(symbol, df, period='daily', adjust='qfq'):
    """保存为标准 CSV 到 data 目录"""
    _, period_tag, _ = KLINE_PERIOD_MAP[period]
    _, adjust_tag, _ = ADJUST_MAP[adjust]
    filename = f'{symbol}_{period_tag}_{adjust_tag}.csv'
    filepath = os.path.join(get_data_dir(), filename)
    df.to_csv(filepath, index=False)
    return filepath, filename


def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = get_data_dir()
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "不支持的文件格式"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"缺少必需列: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"文件加载失败: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0 and len(prediction_results) > 0:
            last_pred = prediction_results[0]
            first_actual = actual_data[0]
            save_data['analysis']['continuity'] = {
                'last_prediction': {
                    'open': last_pred['open'],
                    'high': last_pred['high'],
                    'low': last_pred['low'],
                    'close': last_pred['close']
                },
                'first_actual': {
                    'open': first_actual['open'],
                    'high': first_actual['high'],
                    'low': first_actual['low'],
                    'close': first_actual['close']
                },
                'gaps': {
                    'open_gap': abs(last_pred['open'] - first_actual['open']),
                    'high_gap': abs(last_pred['high'] - first_actual['high']),
                    'low_gap': abs(last_pred['low'] - first_actual['low']),
                    'close_gap': abs(last_pred['close'] - first_actual['close'])
                },
                'gap_percentages': {
                    'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                    'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                    'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                    'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                }
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def generate_analysis_conclusion(x_df, prediction_results, actual_data, lookback, pred_len, has_comparison):
    """根据预测结果与可选的真实对比数据，生成中文分析结论"""
    last_input_close = float(x_df['close'].iloc[-1])
    pred_closes = [float(p['close']) for p in prediction_results]
    first_pred = pred_closes[0]
    last_pred = pred_closes[-1]
    pred_max = max(pred_closes)
    pred_min = min(pred_closes)

    change_pct = (last_pred - last_input_close) / last_input_close * 100
    short_change = (first_pred - last_input_close) / last_input_close * 100

    if change_pct > 5:
        trend = '偏多'
    elif change_pct < -5:
        trend = '偏空'
    else:
        trend = '震荡'

    points = [
        f'输入窗口（{lookback} 根 K 线）最后一根收盘价：{last_input_close:.2f} 元',
        f'预测首日收盘约 {first_pred:.2f} 元，相对输入末期 {short_change:+.1f}%',
        f'预测 {pred_len} 根 K 线收盘价区间：{pred_min:.2f} ~ {pred_max:.2f} 元',
        f'预测期末收盘约 {last_pred:.2f} 元，相对输入末期 {change_pct:+.1f}%',
    ]

    metrics = {
        'last_input_close': round(last_input_close, 2),
        'first_pred_close': round(first_pred, 2),
        'last_pred_close': round(last_pred, 2),
        'change_pct': round(change_pct, 2),
        'trend': trend,
    }

    if has_comparison and actual_data:
        min_len = min(len(prediction_results), len(actual_data))
        mae = rmse = mape = 0.0
        direction_hits = 0

        for i in range(min_len):
            pred_close = float(prediction_results[i]['close'])
            act_close = float(actual_data[i]['close'])
            err = abs(pred_close - act_close)
            mae += err
            rmse += err * err
            mape += (err / act_close) * 100 if act_close else 0

            if i > 0:
                pred_dir = pred_close - float(prediction_results[i - 1]['close'])
                act_dir = act_close - float(actual_data[i - 1]['close'])
                if (pred_dir >= 0 and act_dir >= 0) or (pred_dir < 0 and act_dir < 0):
                    direction_hits += 1

        mae /= min_len
        rmse = (rmse / min_len) ** 0.5
        mape /= min_len
        direction_rate = (direction_hits / (min_len - 1) * 100) if min_len > 1 else 0

        actual_last = float(actual_data[-1]['close'])
        actual_change = (actual_last - last_input_close) / last_input_close * 100

        if mape < 5:
            fit_quality = '拟合较好'
        elif mape < 15:
            fit_quality = '存在一定偏差'
        else:
            fit_quality = '偏差较大，仅供参考'

        points.append(f'回测对比：MAE {mae:.2f} 元，RMSE {rmse:.2f} 元，MAPE {mape:.1f}%（{fit_quality}）')
        if min_len > 1:
            points.append(f'涨跌方向命中率约 {direction_rate:.0f}%（相邻 K 线收盘方向）')
        points.append(f'真实期末收盘 {actual_last:.2f} 元，实际涨跌幅 {actual_change:+.1f}%')

        pred_vs_actual = last_pred - actual_last
        if abs(pred_vs_actual) / actual_last * 100 < 10:
            points.append('预测期末价与真实期末价接近，模型在该窗口内整体量级尚可')
        elif pred_vs_actual > 0:
            points.append('预测期末价高于真实期末价，模型在该段可能偏乐观')
        else:
            points.append('预测期末价低于真实期末价，模型在该段可能偏保守')

        metrics.update({
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'direction_rate': round(direction_rate, 1),
            'actual_last_close': round(actual_last, 2),
            'actual_change_pct': round(actual_change, 2),
        })
        summary = (
            f'综合判断：本窗口为历史回测模式。模型对后续 {pred_len} 根 K 线给出「{trend}」路径，'
            f'与真实数据平均误差 MAPE {mape:.1f}%。'
        )
    else:
        summary = (
            f'综合判断：模型基于最近 {lookback} 根 K 线，对后续 {pred_len} 根给出「{trend}」路径参考，'
            f'预测期末较输入末期约 {change_pct:+.1f}%。'
        )

    return {
        'summary': summary,
        'points': points,
        'disclaimer': '以上结论由 Kronos 模型自动生成，仅供研究参考，不构成任何投资建议。',
        'metrics': metrics,
    }

def build_candle_hovertext(frame, x_series):
    """生成 K 线中文悬停文本（兼容 Plotly 5.x，不使用 hovertemplate）"""
    texts = []
    for i in range(len(frame)):
        ts = x_series.iloc[i] if hasattr(x_series, 'iloc') else x_series[i]
        if hasattr(ts, 'strftime'):
            ts_str = ts.strftime('%Y-%m-%d %H:%M')
        else:
            ts_str = str(ts)
        row = frame.iloc[i]
        texts.append(
            f"时间: {ts_str}<br>"
            f"开盘: {float(row['open']):.4f}<br>"
            f"最高: {float(row['high']):.4f}<br>"
            f"最低: {float(row['low']):.4f}<br>"
            f"收盘: {float(row['close']):.4f}"
        )
    return texts


def format_user_error(error):
    """将异常转为简短中文提示，避免把 Plotly 长文档返回给前端"""
    msg = str(error)
    if 'Invalid property specified' in msg and 'plotly' in msg.lower():
        first_line = msg.split('\n')[0]
        return f'图表配置错误: {first_line}'
    if len(msg) > 400:
        return msg[:400] + '...'
    return msg


def infer_kline_freq(timestamps):
    """推断 K 线时间间隔（中位数），避免 CSV 首两行上市空隙导致间隔失真"""
    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(days=1)
    return diffs.median()


def select_prediction_window(df, lookback, pred_len, start_row=None, start_date=None):
    """选取固定长度窗口：前 lookback 根用于预测，后 pred_len 根用于对比"""
    window_size = lookback + pred_len
    if len(df) < window_size:
        return None, None, f'数据长度不足，至少需要 {window_size} 根 K 线，当前仅有 {len(df)} 根'

    if start_row is not None:
        start_row = int(start_row)
        if start_row < 0 or start_row + window_size > len(df):
            return None, None, (
                f'窗口起始行 {start_row} 无效，有效范围 0～{len(df) - window_size}'
            )
        time_range_df = df.iloc[start_row:start_row + window_size].copy()
        return time_range_df, start_row, None

    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        matched = df[mask]
        if len(matched) < window_size:
            return None, None, (
                f'从起始时间 {start_dt.strftime("%Y-%m-%d %H:%M")} 起数据不足，'
                f'至少需要 {window_size} 根 K 线，当前仅有 {len(matched)} 根'
            )
        time_range_df = matched.iloc[:window_size].copy()
        return time_range_df, int(time_range_df.index[0]), None

    # 默认：窗口右端对齐最新一根 K 线
    start_row = len(df) - window_size
    time_range_df = df.iloc[start_row:start_row + window_size].copy()
    return time_range_df, start_row, None


def create_prediction_chart(
    df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0, pred_timestamps=None
):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    hist_x = historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index

    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=hist_x,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='历史数据（400 根 K 线）',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350',
        hovertext=build_candle_hovertext(historical_df, hist_x),
        hoverinfo='text',
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # 优先使用模型预测时传入的 y_timestamp，保证与对比表日期一致
        if pred_timestamps is not None and len(pred_timestamps) == len(pred_df):
            pred_ts = pd.to_datetime(pred_timestamps).reset_index(drop=True)
        elif 'timestamps' in df.columns and len(historical_df) > 0:
            last_timestamp = historical_df['timestamps'].iloc[-1]
            freq = infer_kline_freq(df['timestamps'])
            pred_ts = pd.date_range(
                start=last_timestamp + freq,
                periods=len(pred_df),
                freq=freq
            )
        else:
            pred_ts = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_ts,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='模型预测（120 根 K 线）',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043',
            hovertext=build_candle_hovertext(pred_df, pd.Series(pred_ts)),
            hoverinfo='text',
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # 真实对比段直接使用 CSV 中的交易日时间戳
        if 'timestamps' in actual_df.columns:
            actual_timestamps = actual_df['timestamps']
        elif 'pred_ts' in locals():
            actual_timestamps = pred_ts
        elif len(historical_df) > 0 and 'timestamps' in df.columns:
            last_timestamp = historical_df['timestamps'].iloc[-1]
            freq = infer_kline_freq(df['timestamps'])
            actual_timestamps = pd.date_range(
                start=last_timestamp + freq,
                periods=len(actual_df),
                freq=freq
            )
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='真实数据（120 根 K 线）',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336',
            hovertext=build_candle_hovertext(actual_df, pd.Series(actual_timestamps)),
            hoverinfo='text',
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos 预测结果：400 根历史 K 线 + 120 根预测 vs 120 根真实对比',
        xaxis_title='时间',
        yaxis_title='价格',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(title=dict(text='图例')),
        hovermode='x unified',
        hoverlabel=dict(align='left'),
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)


@app.route('/api/fetch-stock', methods=['POST'])
def fetch_stock_data():
    """根据股票代码在线拉取 K 线并保存到 data 目录"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', '').strip()
        period = data.get('period', 'daily')
        adjust = data.get('adjust', 'qfq')

        df = fetch_a_share_kline(symbol, period=period, adjust=adjust)
        if len(df) < 520:
            return jsonify({
                'error': f'数据仅 {len(df)} 根 K 线，Web UI 至少需要 520 根（400+120），请换更长历史或改用日 K'
            }), 400

        filepath, filename = save_stock_csv(symbol, df, period=period, adjust=adjust)
        latest_close = float(df['close'].iloc[-1])
        latest_date = df['timestamps'].iloc[-1].strftime('%Y-%m-%d')

        return jsonify({
            'success': True,
            'message': f'已下载 {symbol}，共 {len(df)} 根 K 线，最新 {latest_date} 收盘 {latest_close:.2f} 元',
            'file': {
                'name': filename,
                'path': filepath,
                'size': f"{os.path.getsize(filepath) / 1024:.1f} KB",
            },
            'summary': {
                'symbol': symbol,
                'rows': len(df),
                'start_date': df['timestamps'].min().isoformat(),
                'end_date': df['timestamps'].max().isoformat(),
                'latest_close': latest_close,
            }
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except ConnectionError as e:
        return jsonify({'error': str(e)}), 502
    except requests.RequestException as e:
        return jsonify({'error': f'网络请求失败: {e}'}), 502
    except Exception as e:
        return jsonify({'error': f'下载失败: {e}'}), 500


@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': '文件路径不能为空'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "未知"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "未知"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} 秒"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} 分钟"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} 小时"
            else:
                return f"{avg_diff.days} 天"
        
        # Return data information
        timestamp_list = []
        if 'timestamps' in df.columns:
            timestamp_list = df['timestamps'].dt.strftime('%Y-%m-%d').tolist()

        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'timestamps': timestamp_list,
            'file_name': os.path.basename(file_path),
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'数据加载成功，共 {len(df)} 行'
        })
        
    except Exception as e:
        return jsonify({'error': f'数据加载失败: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': '文件路径不能为空'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'数据长度不足，至少需要 {lookback} 行'}), 400
        
        start_date = data.get('start_date')
        start_row = data.get('start_row')

        time_range_df, historical_start_idx, window_error = select_prediction_window(
            df, lookback, pred_len, start_row=start_row, start_date=start_date
        )
        if window_error:
            return jsonify({'error': window_error}), 400

        window_start_ts = time_range_df['timestamps'].iloc[0]
        window_end_ts = time_range_df['timestamps'].iloc[lookback + pred_len - 1]
        time_span = window_end_ts - window_start_ts
        prediction_type = (
            f'Kronos 模型预测（窗口第 {historical_start_idx}～'
            f'{historical_start_idx + lookback + pred_len - 1} 行，'
            f'前 {lookback} 根预测、后 {pred_len} 根对比，跨度 {time_span}）'
        )

        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')

                x_df = time_range_df.iloc[:lookback][required_cols]
                x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                y_timestamp = time_range_df.iloc[lookback:lookback + pred_len]['timestamps']

                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')

                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )

            except Exception as e:
                return jsonify({'error': f'Kronos 模型预测失败: {format_user_error(e)}'}), 500
        else:
            return jsonify({'error': '请先加载 Kronos 模型'}), 400

        # 对比段真实 K 线（与 y_timestamp 同一时间段）
        actual_data = []
        actual_df = time_range_df.iloc[lookback:lookback + pred_len]
        for _, row in actual_df.iterrows():
            actual_data.append({
                'timestamp': row['timestamps'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })

        chart_json = create_prediction_chart(
            df,
            pred_df,
            lookback,
            pred_len,
            actual_df,
            historical_start_idx,
            pred_timestamps=y_timestamp.reset_index(drop=True),
        )

        # 预测结果时间戳直接使用 y_timestamp，与真实对比段逐日对齐
        prediction_results = []
        y_ts_list = y_timestamp.reset_index(drop=True)
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ts = y_ts_list.iloc[i] if i < len(y_ts_list) else None
            prediction_results.append({
                'timestamp': ts.isoformat() if ts is not None else f'T{i}',
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest',
                    'start_row': historical_start_idx,
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")

        analysis_conclusion = generate_analysis_conclusion(
            x_df, prediction_results, actual_data, lookback, pred_len, len(actual_data) > 0
        )

        window_info = {
            'start_row': historical_start_idx,
            'input_start': time_range_df['timestamps'].iloc[0].isoformat(),
            'input_end': time_range_df['timestamps'].iloc[lookback - 1].isoformat(),
            'compare_start': time_range_df['timestamps'].iloc[lookback].isoformat(),
            'compare_end': time_range_df['timestamps'].iloc[lookback + pred_len - 1].isoformat(),
            'file_name': os.path.basename(file_path),
        }
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'window_info': window_info,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'analysis_conclusion': analysis_conclusion,
            'message': f'预测完成，已生成 {pred_len} 个预测点' + (f'，含 {len(actual_data)} 个真实数据点用于对比' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'预测失败: {format_user_error(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos 模型库不可用'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'不支持的模型: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'模型加载成功：{model_config["name"]}（{model_config["params"]}），运行设备 {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos 模型已加载，可以使用',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos 模型可用，尚未加载'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos 模型库不可用，请先安装相关依赖'
        })

if __name__ == '__main__':
    print("正在启动 Kronos Web UI...")
    print(f"模型可用: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("提示: 请在页面中点击「加载模型」")
    else:
        print("提示: 将使用模拟数据进行演示")
    
    app.run(debug=True, host='0.0.0.0', port=7070)

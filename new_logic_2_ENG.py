# ==============================================================================
# --- BLOCK 1: IMPORTS ---
# ==============================================================================
import os
import time
import logging
import sys
import requests
import json
import argparse
import threading
import warnings
from collections import deque, defaultdict
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
from typing import Tuple
from datetime import datetime, timedelta

# --- Data Science & Math ---
import numpy as np
import pandas as pd
import numba
from numba import jit
import pandas_ta as ta
from tqdm import tqdm

# --- Binance API ---
from dotenv import load_dotenv
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.error import ClientError

# --- ML / Research Legacy (Preserved for R&D Context) ---
import joblib
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (precision_score, accuracy_score, confusion_matrix, 
                             classification_report, f1_score, precision_recall_curve, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import RobustScaler

# --- Deep Learning (Legacy/Experimental) ---
# Note: Heavy libraries (TF/Keras/CatBoost/XGB) are kept for archive compatibility
try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import tensorflow as tf
    from keras.models import Model
    from keras import layers
    from keras.layers import (Input, MultiHeadAttention, LayerNormalization, Dense, 
                              Dropout, GlobalAveragePooling1D, Add, LSTM)
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import keras.backend as K
except ImportError:
    # Allow execution even if deep learning libs are missing in the lean environment
    pass


# --- MAIN SCRIPT LOGGER ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d UTC %(levelname)-8s %(name)-20s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
main_logger = logging.getLogger(__name__)
# --- END OF BLOCK ---

# ==============================================================================
# --- ВСТАВИТИ ЦЕ ОДРАЗУ ПІСЛЯ ІМПОРТІВ (ПЕРЕД КЛАСОМ TradingBotInstance) ---
# ==============================================================================

class LivePhysicsEngine:
    """
    Генерує фізичні фічі (Гравітація, Стіни, Тиск) в реальному часі.
    Повна копія логіки з lab_research.py для сумісності з Трансформером.
    """
    def __init__(self):
        # Сітка рівнів (Grid Steps) - має співпадати з тренуванням
        self.available_steps = [1e-05, 2.5e-05, 5e-05, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 
                                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 
                                25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]
        # Мультиплікатори масштабу
        self.spectrum_multipliers = [0.2, 0.5, 1.0, 2.5, 5.0]
        # Пам'ять ринку (об'єми на рівнях)
        self.memory = {s: {} for s in self.available_steps}

    def process_window(self, df):
        """
        Генерує фічі для AI.
        ВИПРАВЛЕНО:
        1. base_series (X) -> для побудови стін (Зір Нейронки).
        2. atr_series (Simple High-Low) -> повертається для розрахунку Стопів.
        """
        # 1. Очищення пам'яті
        self.memory = {s: {} for s in self.available_steps}
        features_list = []
        
        # --- А. ОТРИМУЄМО X (ДЛЯ НЕЙРОНКИ) ---
        # Це те, що виправляє "зір" бота. Він має бачити сітку через X.
        if 'x' in df.columns:
            base_series = df['x'].values
        elif 'safe_x' in df.columns:
            base_series = df['safe_x'].values
        else:
            # Fallback (якщо раптом індикатори не порахувались, хоча ми це фіксили)
            # Тут використовуємо ATR як милицю, щоб не впало
            try:
                base_series = df.ta.atr(length=14).fillna(0).values
            except:
                h = df['high'].values
                l = df['low'].values
                c = df['close'].values
                tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
                base_series = pd.Series(tr).rolling(14).mean().fillna(0).values

        # --- Б. ОТРИМУЄМО ATR (ДЛЯ СТОПІВ) ---
        # Валідатор використовував просте (High - Low) SMA 14.
        # Це повертаємо назовні, щоб бот знав, куди ліпити стоп.
        high_s = pd.Series(df['high'].values)
        low_s = pd.Series(df['low'].values)
        
        # Тупо High - Low (без гепів), SMA 14. Як у validator_honest.py
        atr_series = (high_s - low_s).rolling(window=14).mean().fillna(0).values
        
        # --- В. ПІДГОТОВКА ДАНИХ ДЛЯ ЦИКЛУ ---
        close = df['close'].values
        vol = df['volume'].values
        op = df['open'].values
        hi = df['high'].values
        lo = df['low'].values
        
        # 3. Головний цикл
        for i in range(len(df)):
            current_base = base_series[i] # <-- ДЛЯ СІТКИ БЕРЕМО X!
            
            # Якщо X ще не порахувався (початок історії)
            if np.isnan(current_base) or current_base == 0: 
                features_list.append(None)
                continue

            current_price = close[i]
            
            # --- ВАЖЛИВО: Active Base тепер це X ---
            # Ми прив'язуємось до X, бо це "рідна" мова нейронки з навчання
            active_base = min(self.available_steps, key=lambda x: abs(x - current_base))
            
            # --- UPDATE MEMORY ---
            candle_range = hi[i] - lo[i]
            if candle_range == 0: candle_range = 1e-9
            
            pressure_val = (current_price - op[i]) / candle_range * vol[i]
            
            for step in self.available_steps:
                g_idx = int(current_price / step)
                if g_idx not in self.memory[step]: 
                    self.memory[step][g_idx] = {'V': 0.0, 'P': 0.0}
                self.memory[step][g_idx]['V'] += vol[i]
                self.memory[step][g_idx]['P'] += pressure_val

            # --- FEATURES GENERATION ---
            current_feats = []
            focus_steps = []
            for mult in self.spectrum_multipliers:
                target = active_base * mult
                actual_step = min(self.available_steps, key=lambda x: abs(x - target))
                focus_steps.append((mult, actual_step))
            
            total_grav_buy = 0.0
            total_grav_sell = 0.0
            
            for mult, step in focus_steps:
                current_grid_idx = int(round(current_price / step))
                current_grid_price = current_grid_idx * step
                
                mem_main = self.memory[step].get(current_grid_idx, {'V': 0, 'P': 0})
                mem_below = self.memory[step].get(current_grid_idx - 1, {'V': 0, 'P': 0})
                mem_above = self.memory[step].get(current_grid_idx + 1, {'V': 0, 'P': 0})
                
                # 1. Walls
                current_feats.append(np.log1p(mem_main['V']))
                current_feats.append(np.log1p(mem_above['V']))
                current_feats.append(np.log1p(mem_below['V']))
                
                # 2. Pressure
                press = mem_main['P'] / (mem_main['V'] + 1) if mem_main['V'] > 0 else 0
                current_feats.append(press)
                
                # 3. Gravity
                dist_up = ((current_grid_idx + 1) * step) - current_price
                grav_up = mem_above['V'] / (dist_up**2 + 1e-6)
                total_grav_sell += grav_up
                
                dist_down = current_price - ((current_grid_idx - 1) * step)
                grav_down = mem_below['V'] / (dist_down**2 + 1e-6)
                total_grav_buy += grav_down
                
                dist_main = abs(current_price - current_grid_price)
                grav_main = mem_main['V'] / (dist_main**2 + 1e-6)
                if current_price < current_grid_price: total_grav_sell += grav_main
                else: total_grav_buy += grav_main
            
            # Global Feats
            current_feats.append(np.log1p(total_grav_buy))
            current_feats.append(np.log1p(total_grav_sell))
            current_feats.append(np.log1p(total_grav_sell) - np.log1p(total_grav_buy))
            current_feats.append(0.0) # Dummy
            
            features_list.append(current_feats)
            
        # ПОВЕРТАЄМО ДВА МАСИВИ:
        # 1. features_list -> Для AI
        # 2. atr_series -> Для розрахунку SL/TP в _check_and_execute_trade_logic
        return features_list, atr_series



# ==============================================================================
# --- ВСТАВИТИ ПІСЛЯ LivePhysicsEngine І ПЕРЕД TradingBotInstance ---
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        return x.transpose(0, 1)

class PhysicsAwareTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(PhysicsAwareTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 3)
        )

    def forward(self, src):
        x = self.embedding(src)
        x = self.layer_norm(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        last_state = output[:, -1, :] 
        logits = self.classifier(last_state)
        return logits

# ADD THESE TWO LINES AFTER ALL IMPORTS:
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

getcontext().prec = 12
# Define path to directory and .env file
SCRIPT_DIR_NEW_LOGIC = os.path.dirname(os.path.abspath(__file__))
KEY_ENV_PATH_LOGIC = os.path.join(SCRIPT_DIR_NEW_LOGIC, "key.env") # <-- Added
load_dotenv(dotenv_path=KEY_ENV_PATH_LOGIC) # <-- Changed

GLOBAL_STOP_FLAG_FILE_PATH = os.path.join(SCRIPT_DIR_NEW_LOGIC, "STOP_BOT_NOW.flag")
RELOAD_CONFIG_FLAG_FILE_PATH = os.path.join(SCRIPT_DIR_NEW_LOGIC, "RELOAD_CONFIG.flag")
DEFAULT_BOT_CONFIG_FILE = os.path.join(SCRIPT_DIR_NEW_LOGIC, "active_bots_config.json")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_regression_channel(series: pd.Series, lookback: int, std_dev_multiplier: float = 2.0) -> pd.DataFrame:
    """
    VECTORIZED version for linear regression channel calculation.
    Significantly faster than the iterative loop version.
    """
    if len(series) < lookback:
        return pd.DataFrame(index=series.index, columns=['regr_line', 'upper_channel', 'lower_channel', 'slope'])

    # Create time index (x)
    x = np.arange(len(series))
    
    # Use rolling window to calculate sums required for regression
    x_roll = pd.Series(x, index=series.index).rolling(window=lookback)
    y_roll = series.rolling(window=lookback)
    
    sum_x = x_roll.sum()
    sum_y = y_roll.sum()
    sum_xy = (pd.Series(x, index=series.index) * series).rolling(window=lookback).sum()
    sum_x_sq = (pd.Series(x, index=series.index)**2).rolling(window=lookback).sum()
    
    # Formula for slope
    numerator = lookback * sum_xy - sum_x * sum_y
    denominator = lookback * sum_x_sq - sum_x**2
    slope = numerator / denominator
    
    # Formula for intercept
    intercept = (sum_y - slope * sum_x) / lookback
    
    # Calculate regression line values for the end point of each window
    regr_line_values = intercept + slope * (x)
    
    # Calculate standard deviation
    residuals_sq = (series - regr_line_values)**2
    std_dev = np.sqrt(residuals_sq.rolling(window=lookback).sum() / (lookback - 1)) * std_dev_multiplier
    
    # Create DataFrame with results
    result = pd.DataFrame(index=series.index)
    result['regr_line'] = regr_line_values
    result['upper_channel'] = regr_line_values + std_dev
    result['lower_channel'] = regr_line_values - std_dev
    result['slope'] = slope
    
    return result

def rolling_pearson(y_data: np.ndarray) -> float:
    """Calculates Pearson correlation coefficient for a time series."""
    n = len(y_data)
    if n < 2:
        return np.nan
    x_data = np.arange(n)
    # Check if data is constant to avoid errors
    if np.all(y_data == y_data[0]):
        return 0.0
    
    corr_matrix = np.corrcoef(x_data, y_data)
    return corr_matrix[0, 1]

def get_all_historical_futures_data(symbol, interval, start_str=None, end_str=None, lookback_months=None):
    """
    LOADS HISTORICAL DATA for the specified period or the last N months.
    The lookback_months parameter takes precedence.
    
    ★★★ VERSION FIXED FOR 1-TO-1 VALIDATION WITH REAL BOT ★★★
    1. Removes 'utc=True' (matches bot behavior).
    2. Adds 'keep=last' in drop_duplicates (matches bot behavior).
    3. !!! ADDS COLUMN TRUNCATION (matches bot behavior) !!!
    """
    load_dotenv(dotenv_path="key.env")
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = UMFutures(key=api_key, secret=api_secret)

    if lookback_months and lookback_months > 0:
        end_dt_calc = datetime.now()
        start_dt_calc = end_dt_calc - timedelta(days=int(lookback_months * 30.5))
        start_str = start_dt_calc.strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"TRAINING: Using lookback_months={lookback_months}. Calculated start date: {start_str}")
    elif not start_str:
        raise ValueError("Must provide either start_str or lookback_months")

    logging.info(f"TRAINING: Loading full history: {symbol} ({interval}) from {start_str} to {end_str or 'current time'}...")
    start_ts = int(datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000) if not end_str else int(datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

    all_klines = []
    current_start_ts = start_ts

    try:
        disable_progress = args.no_progress_bar
    except NameError:
        disable_progress = False

    with tqdm(total=(end_ts - start_ts), unit='ms', desc=f"Loading {symbol}", disable=disable_progress) as pbar:
        while current_start_ts < end_ts:
            klines = client.klines(symbol=symbol, interval=interval, startTime=int(current_start_ts), limit=1500)
            if not klines:
                break
            all_klines.extend(klines)
            last_kline_ts = klines[-1][0]
            pbar.update(last_kline_ts - current_start_ts + 1)
            if last_kline_ts >= end_ts:
                break
            current_start_ts = last_kline_ts + 1
            time.sleep(0.3)

    if not all_klines: return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # --- ★★★ FINAL FIX (1-TO-1 MATCH WITH BOT) ★★★ ---
    # Truncate DF to 5 columns + timestamp, AS DONE IN _get_klines_df_rest
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    # --- ★★★ END OF FIX ★★★ ---

    # --- (Previous fixes) ---
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str) if end_str else pd.to_datetime(datetime.now())
    # --- (End of previous fixes) ---

    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]

    # This 'for' loop is redundant due to 'astype(float)', 
    # but keeping it for consistency.
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('timestamp', inplace=True)
    return df

@numba.njit
def find_closest_step(value, step_grid):
    idx = np.abs(step_grid - value).argmin()
    return step_grid[idx]

@numba.jit(nopython=True, nogil=True)
def calculate_step_level_numba(close_arr, rounded_close_arr, x_arr):
    n = len(close_arr)
    if n == 0: return np.zeros(n) * np.nan
    step_level_out = np.full(n, np.nan)
    prev_level = np.nan
    start_index = -1

    for i in range(n):
        if not np.isnan(rounded_close_arr[i]) and not np.isnan(x_arr[i]) and x_arr[i] != 0:
            prev_level = rounded_close_arr[i]
            for j in range(i + 1):
                 step_level_out[j] = prev_level
            start_index = i + 1
            break

    if start_index == -1:
        return step_level_out

    for i in range(start_index, n):
        rounded = rounded_close_arr[i]
        x_val = x_arr[i]
        close_val = close_arr[i]
        current_level = prev_level

        if not (np.isnan(close_val) or np.isnan(x_val) or np.isnan(prev_level) or x_val == 0):
            if close_val >= prev_level + x_val:
                current_level = rounded if not np.isnan(rounded) else prev_level + x_val
            elif close_val <= prev_level - x_val:
                current_level = rounded if not np.isnan(rounded) else prev_level - x_val

        if np.isnan(current_level):
            current_level = prev_level

        step_level_out[i] = current_level
        prev_level = current_level
    return step_level_out

@numba.njit
def calculate_step_level_numba_hl(highs, lows, closes, atrs, step_grid, start_price):
    n = len(closes)
    step_level_out = np.zeros(n, dtype=np.float64)
    
    # 1. Determine first step
    current_step_val = find_closest_step(atrs[0], step_grid)
    
    # !!! FIX: Round start price to nearest grid level !!!
    # Was: current_level = start_price
    # Now:
    current_level = round(start_price / current_step_val) * current_step_val
    
    step_level_out[0] = current_level

    for i in range(1, n):
        # Thresholds for switching
        up_threshold = current_level + current_step_val
        down_threshold = current_level - current_step_val
        
        # Check breakout logic: first check, then update
        
        if highs[i] >= up_threshold:
            # UP Breakout
            # 1. Update step for new volatility
            current_step_val = find_closest_step(atrs[i], step_grid)
            
            # 2. Calculate new level.
            # Instead of simple addition, we snap to nearest grid based on up_threshold.
            # Variant for strict grid:
            current_level = round(up_threshold / current_step_val) * current_step_val
            
        elif lows[i] <= down_threshold:
            # DOWN Breakout
            current_step_val = find_closest_step(atrs[i], step_grid)
            
            # Snap level
            current_level = round(down_threshold / current_step_val) * current_step_val
            
        # If no breakout, current_level remains unchanged
        step_level_out[i] = current_level

    return step_level_out

@numba.jit(nopython=True, nogil=True)
def calculate_step_index(step_level_arr: np.ndarray) -> np.ndarray:
    """
    Corrected version based on user logic.
    Increments counter at start of new trend and continuation.
    Resets counter to 1 on trend reversal.
    Maintains counter value if level is unchanged.
    """
    n = len(step_level_arr)
    if n == 0:
        return np.zeros(n, dtype=np.int64)

    step_indices = np.zeros(n, dtype=np.int64)
    trend_dir = 0    # 0 = Undefined, 1 = Up, -1 = Down
    step_index = 0

    for i in range(n):
        if i == 0:
            step_index = 1
            step_indices[i] = step_index
            # Direction on first step is undefined
            continue

        # Determine direction of current move
        current_dir = 0
        if step_level_arr[i] > step_level_arr[i-1]:
            current_dir = 1
        elif step_level_arr[i] < step_level_arr[i-1]:
            current_dir = -1

        # If no move (flat), counter and trend direction do not change
        if current_dir == 0:
            step_indices[i] = step_index
            continue
        
        # If trend is just starting (was undefined) or continuing
        if trend_dir == 0 or current_dir == trend_dir:
            step_index += 1
        # Else - trend reversal occurred
        else:
            step_index = 1
        
        # Update state for next iteration
        trend_dir = current_dir
        step_indices[i] = step_index
        
    return step_indices

@numba.jit(nopython=True, nogil=True)
def _calculate_pivot_step_index_price_based(step_level_arr: np.ndarray, x_arr: np.ndarray, reset_threshold: int = 3) -> np.ndarray:
    """
    RED INDEX (V8 - PIVOT SWITCH).
    Logic implementation:
    1. If pullback from High >= 3 steps -> Trend becomes DOWN, Pivot becomes High.
    2. Now index = (PivotHigh - Price) / X.
    3. E.g. move 1090 (High) -> 1060 index = 3.
    4. Bounce 1060 -> 1070 index = 2 (closer to High).
    """
    n = len(step_level_arr)
    pivot_step_index_out = np.zeros(n, dtype=np.int64)
    
    if n == 0: return pivot_step_index_out

    # 1 = UP Trend (Measure from Low), -1 = DOWN Trend (Measure from High)
    # Default start as UP, pivot is start
    wave_direction = 1 
    pivot_price = step_level_arr[0] # This is our "Zero point"
    
    # Variable to track extrema within current wave (to catch reversal)
    current_extreme = step_level_arr[0]

    for i in range(n):
        current_price = step_level_arr[i]
        current_x = x_arr[i]
        
        if np.isnan(current_price) or np.isnan(current_x) or current_x == 0:
            pivot_step_index_out[i] = 0
            continue
        
        if i == 0:
            pivot_step_index_out[i] = 0
            continue

        # Update extrema of current wave
        if wave_direction == 1: # If going UP
            if current_price > current_extreme:
                current_extreme = current_price
        else: # If going DOWN
            if current_price < current_extreme:
                current_extreme = current_price

        # Reversal Check (Trigger)
        # If UP, but price fell from Max by >= 3 steps
        if wave_direction == 1:
            dist_from_high = (current_extreme - current_price) / current_x
            if dist_from_high >= reset_threshold:
                # SWITCH! Now falling.
                wave_direction = -1
                pivot_price = current_extreme # New zero is High
                current_extreme = current_price # Reset local extreme

        # If DOWN, but price rose from Min by >= 3 steps
        elif wave_direction == -1:
            dist_from_low = (current_price - current_extreme) / current_x
            if dist_from_low >= reset_threshold:
                # SWITCH! Now rising.
                wave_direction = 1
                pivot_price = current_extreme # New zero is Low
                current_extreme = current_price

        # FINAL CALCULATION
        # Index is always distance from Actual Pivot (High or Low)
        # round() used so 2.99 becomes 3
        raw_distance = abs(current_price - pivot_price) / current_x
        pivot_step_index_out[i] = int(round(raw_distance))

    return pivot_step_index_out

# --- Telegram Logging Handler ---
class TelegramLoggingHandler(logging.Handler):
    def __init__(self, token, chat_id_str, symbol_prefix=""):
        super().__init__()
        self.token = token
        self.chat_id = None
        self.symbol_prefix = f"[{symbol_prefix}] " if symbol_prefix else ""
        if chat_id_str:
            try:
                self.chat_id = int(chat_id_str)
            except ValueError:
                print(f"CONSOLE ERROR: Invalid TELEGRAM_CHAT_ID: {chat_id_str}")
        if not self.token:
            print("CONSOLE ERROR: Token for Telegram not provided.")

    def emit(self, record):
        if not self.token or not self.chat_id:
            return
            
        log_entry = self.format(record)
        
        # --- START OF NEW FILTER ---

        if "ORDER_TRADE_UPDATE for" in log_entry:
            return

        # 2. Define keywords that MUST be sent
        essential_keywords = [
            "!!!",                  # SL/TP Execution
            "filled",               # Entry confirmation
            "action: new entry",    # New limit order confirmation
            "critical",             # Critical errors
            "error",                # Errors
            "failed",               # Failures
            "unprotected",          # Position without protection
            "stopped due to",       # Bot stop reason
            "shut down",            # Shutdown (was: завершив роботу)
            "signal rejected",      # Signal rejection (was: сигнал відхилено)
            "ML_FILTER",
            "LIVE STATS",           # Pass stats table
            "ANALYSIS",             # Pass pattern analysis
            "TRAP SNAP",
            "PnL",
            "Balance",
            "Delta",
            "WIN",
            "LOSS"                  # Trap trigger backup
        ]

        # Check if message should be sent
        should_send = record.levelno >= logging.WARNING or \
                      any(keyword.lower() in log_entry.lower() for keyword in essential_keywords)

        if not should_send:
            return 
            
        # --

        log_entry_with_prefix = f"[{self.symbol_prefix}] {log_entry}"

        if len(log_entry_with_prefix) > 4096:
            log_entry_with_prefix = log_entry_with_prefix[:4090] + "\n... (truncated)"
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': f"[{self.symbol_prefix}] {log_entry}", # Simply join string without Markdown
            # 'parse_mode': None # SET TO NONE OR REMOVE
        }
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            # self.logger.debug(f"Telegram Handler: Message sent successfully. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            # Log full response text here if available
            error_details = f"Error sending log to Telegram (requests): {e}"
            if hasattr(e, 'response') and e.response is not None:
                error_details += f", Response status: {e.response.status_code}, Response text: {e.response.text}"
            print(f"CONSOLE ERROR (TelegramLoggingHandler): {error_details}")
            # Use a basic logger for internal errors to avoid recursion, or just print
            logging.error(f"Telegram Logging Error (from handler): {error_details}", exc_info=True)
        except Exception as e_unknown:
            print(f"CONSOLE ERROR (TelegramLoggingHandler): Unknown error sending log to Telegram: {e_unknown}")
            # Use a basic logger for internal errors to avoid recursion, or just print
            logging.error(f"Telegram Logging Unknown Error (from handler): {e_unknown}", exc_info=True)

# --- TradingBotInstance ---
class TradingBotInstance:
    def __init__(self, symbol: str, interval: str, api_key: str, api_secret: str,
                 telegram_bot_token: str = None, telegram_chat_id_str: str = None,
                 bot_config: dict = None, orchestrator_stop_event: threading.Event = None, is_backtest: bool = False):
        
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        self.symbol = symbol
        self.interval = interval
        self.api_key = api_key
        self.api_secret = api_secret

        self.telegram_bot_token_for_logging = telegram_bot_token
        self.telegram_chat_id_for_logging_str = telegram_chat_id_str
        self.config = bot_config if bot_config else {}
        self.orchestrator_stop_event = orchestrator_stop_event if orchestrator_stop_event else threading.Event()
        self.step_coeff = 0.5 

        # --- LOGGER SETUP ---
        self.logger = logging.getLogger(f"TradingBot.{self.symbol}")
        if not self.logger.handlers:
            log_level_str = self.config.get('LOG_LEVEL', 'INFO').upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            self.logger.setLevel(log_level)
            log_filename = os.path.join(SCRIPT_DIR, f"trading_bot_{self.symbol}_sessions.log")
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d UTC %(levelname)-8s %(name)-15s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(log_level)
            self.logger.addHandler(stream_handler)
            if self.telegram_bot_token_for_logging and self.telegram_chat_id_for_logging_str:
                tg_handler = TelegramLoggingHandler(self.telegram_bot_token_for_logging, self.telegram_chat_id_for_logging_str, symbol_prefix=self.symbol)
                tg_handler.setLevel(logging.INFO)
                self.logger.addHandler(tg_handler)

        self.logger.info(f"Initializing TradingBotInstance for {self.symbol}")

        # --- PHYSICS & TRANSFORMER INIT ---
        self.model_path = os.path.join(SCRIPT_DIR, "best_transformer_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ініціалізація рушія фізики
        self.physics_engine = LivePhysicsEngine()
        self.ai_model = None

        try:
            # INPUT_DIM = 23 (5 масштабів * 4 фічі + 3 глобальні гравітації)
            # Параметри мають співпадати з train_transformer_feb.py
            # ЗМІНИВ 23 НА 24
            self.ai_model = PhysicsAwareTransformer(input_dim=24, d_model=128, nhead=4, num_layers=3, dropout=0.1).to(self.device)
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            self.ai_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.ai_model.eval()
            self.logger.info(f"🧠 PHYSICS TRANSFORMER LOADED: {self.model_path} on {self.device}")
        except Exception as e:
            self.logger.critical(f"❌ FAILED TO LOAD AI MODEL: {e}")
            sys.exit(1) # Зупиняємо бота, якщо мозку немає
            
        # Налаштування Снайпера (твої перевірені)
        self.CONFIDENCE_THRESHOLD = 0.70
        self.SL_ATR_MULT = 1.0
        self.TP_ATR_MULT = 3.0
        
        # Старі змінні для сумісності (щоб решта коду не впала)
        self.ai_stop_loss_pct = 0.01 
        self.ai_take_profit_pct = 0.03

        # --- FINANCIAL PARAMS ---
        self.initial_capital_per_trade_usdt = Decimal(str(self.config.get('INITIAL_CAPITAL_PER_TRADE_USDT', '1.0')))
        self.virtual_bot_capital_usdt = Decimal(str(self.config.get('VIRTUAL_BOT_CAPITAL_USDT', '1.0')))
        self.bot_starting_virtual_capital = self.virtual_bot_capital_usdt
        # DEFAULT TO 0.3 (30% Loss / 70% Equity Retained)
        self.bot_virtual_capital_stop_loss_percent = Decimal('0.70') 
        self.total_account_stop_loss_usdt = Decimal(str(self.config.get('TOTAL_ACCOUNT_STOP_LOSS_USDT', '0.0')))
        
        self.min_distance_from_entry_ticks = Decimal(str(self.config.get('MIN_DISTANCE_FROM_ENTRY_TICKS', '5')))
        self.liquidation_price_buffer_ticks = Decimal(str(self.config.get('LIQUIDATION_PRICE_BUFFER_TICKS', '10')))
        self.max_risk_percent_of_per_trade_capital = Decimal(str(self.config.get('MAX_RISK_PERCENT_OF_PER_TRADE_CAPITAL', '0.7')))
        
        # --- CRITICAL MISSING PARAMS ---
        self.residual_smooth = int(self.config.get('RESIDUAL_SMOOTH', 10))
        self.target_percent_param = float(self.config.get('TARGET_PERCENT_PARAM', 20.0))
        self.tp_coeff = Decimal('1.0') 
        self.sl_coeff = Decimal('0.9')

        self.atr_period = int(self.config.get('ATR_PERIOD', 14))
        self.selected_timeframe_higher = self.config.get('SELECTED_TIMEFRAME_HIGHER', '2h')
        self.limit_main_tf = int(self.config.get('LIMIT_MAIN_TF', 1500))
        self.limit_higher_tf = int(self.config.get('LIMIT_HIGHER_TF', 1500))
        self.step_coeff = float(self.config.get('STEP_COEFF', 0.5))

        self.client = None 
        self.ws_client = None 
        self.listen_key = None
        self.klines_df_main = pd.DataFrame()
        self.klines_df_higher_tf = pd.DataFrame()
        self.last_processed_kline_open_time = None
        self.symbol_trading_rules = {}
        self.leverage_brackets_for_symbol = []
        
        self.current_position_on_exchange = None
        self.current_position_entry_price = Decimal('0.0')
        self.current_position_quantity = 0.0
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.initial_account_balance_usdt = Decimal('0.0')
        self.stop_bot_threshold_balance_usdt = Decimal('0.0')
        self.bot_stopped_due_to_total_sl = False 
        self.should_stop_bot_flag_instance = False
        self.reconnect_websocket_flag = False
        self.last_listen_key_renewal_time = 0
        self.stop_flag_file_path_instance = os.path.join(SCRIPT_DIR, f"STOP_TRADING_BOT_{self.symbol}.flag")
        self.last_heartbeat_log_time = 0
        self.start_time = time.time()
        self.is_processing_position_close = False 
        
        # Trap variables
        self.trap_order_up = None
        self.trap_order_down = None
        self.active_sequence_mode = None
        self.sequence_steps_remaining = 0
        self.behavior_window = deque(maxlen=20)
        self.active_limit_entry_order_details = {}
        self.last_analyzed_step_level = None
        
        # --- FIX: CRITICAL VARIABLES FOR SYNC & BE ---
        self.last_trade_candle_time = None 
        self.real_be_buffer = 0.0015  # <--- ДОДАНО! 0.15% буфер для БУ
        self.ignore_sync_timestamp = 0 # <--- ДОДАНО! Таймер імунітету

        self.logger.info("---------------------------------------------------------------------")
        self.logger.info(f"NEW QUANTUM SESSION for {self.symbol}")
        self.logger.info("---------------------------------------------------------------------")

        self._initialize_binance_client()

                
    def _reset_active_limit_entry_order_details(self):
        # --- NEW TRAP VARIABLES ---
        self.trap_order_up = None   # Dictionary for upper order
        self.trap_order_down = None # Dictionary for lower order
        
        # Reset legacy variable to empty dict to avoid AttributeError
        self.active_limit_entry_order_details = {}

    def _initialize_binance_client(self):
        if not self.api_key or not self.api_secret:
            self.logger.error("API key or secret not provided for Binance client initialization.")
            raise ValueError("Missing API keys.")
        try:
            self.client = UMFutures(key=self.api_key, secret=self.api_secret)
            self.logger.info("Binance UMFutures client (LIVE MARKET) initialized successfully.")
            self.client.ping()
            self.logger.info("Binance API connection successful (ping).")
            server_time = self.client.time()
            self.logger.info(f"Binance Server Time: {pd.to_datetime(server_time['serverTime'], unit='ms')}")
        except ClientError as e:
            self.logger.error(f"Binance API Client Error during initialization: Status {e.status_code}, Code {e.error_code}, Message: {e.error_message}")
            raise ConnectionError(f"Failed to initialize Binance client due to API error: {e.error_message}")
        except Exception as e:
            self.logger.error(f"General Binance client initialization error: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Binance client due to general error: {e}")

    def _send_final_telegram_message(self, message: str):
        if self.telegram_bot_token_for_logging and self.telegram_chat_id_for_logging_str:
            chat_id_int = None
            try:
                chat_id_int = int(self.telegram_chat_id_for_logging_str)
            except ValueError:
                self.logger.error("Invalid Chat ID for final Telegram message.")
                return

            final_message_with_prefix = f"[{self.symbol}] {message}"
            url = f"https://api.telegram.org/bot{self.telegram_bot_token_for_logging}/sendMessage"
            payload = { 'chat_id': chat_id_int, 'text': final_message_with_prefix, 'parse_mode': 'Markdown'}
            try:
                response = requests.post(url, data=payload, timeout=10)
                response.raise_for_status()
                self.logger.info("Final status message sent to Telegram.")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error sending final Telegram message: {e}")
            except Exception as e_unknown:
                self.logger.error(f"Unknown error sending final Telegram message: {e_unknown}", exc_info=True)
        else:
            self.logger.warning("Telegram token or Chat ID not configured for final message.")


    def _get_symbol_trading_rules(self, retries: int = 3, delay: int = 5):
        for attempt in range(retries):
            try:
                self.logger.info(f"Requesting exchangeInfo (Attempt {attempt + 1}/{retries}) for {self.symbol}...")
                exchange_info = self.client.exchange_info()
                for s_info in exchange_info['symbols']:
                    if s_info['symbol'] == self.symbol:
                        rules = {
                            "symbol": self.symbol, "status": s_info.get('status'),
                            "quantityPrecision": int(s_info.get('quantityPrecision', 0)),
                            "pricePrecision": int(s_info.get('pricePrecision', 0)),
                            "orderTypes": s_info.get('orderTypes', []),
                            "minQty": 0.0, "stepSize": "0.00000001",
                            "minNotional": 0.0, "tickSize": "0.00000001",
                            "maxQty": "1000000"
                        }
                        for f_filter in s_info.get('filters', []):
                            if f_filter['filterType'] == 'LOT_SIZE':
                                rules["minQty"] = float(f_filter.get('minQty', "0"))
                                rules["maxQty"] = f_filter.get('maxQty', "1000000")
                                rules["stepSize"] = f_filter.get('stepSize', "0.00000001")
                            elif f_filter['filterType'] == 'MIN_NOTIONAL':
                                rules["minNotional"] = float(f_filter.get('notional', f_filter.get('minNotional', "0")))
                            elif f_filter['filterType'] == 'PRICE_FILTER':
                                rules["tickSize"] = f_filter.get('tickSize', "0.00000001")
                        
                        self.logger.info(f"Trading rules received for {self.symbol}: "
                                         f"qtyPrec={rules['quantityPrecision']}, pricePrec={rules['pricePrecision']}, "
                                         f"minQty={rules['minQty']}, stepSize={rules['stepSize']}, "
                                         f"minNotional={rules['minNotional']}, tickSize={rules['tickSize']}")
                        self.symbol_trading_rules = rules 
                        return True 
                self.logger.warning(f"Symbol {self.symbol} not found in exchangeInfo (Attempt {attempt + 1}/{retries}).")
                if attempt < retries - 1: time.sleep(delay)
                else: return False
            except ClientError as e:
                self.logger.error(f"API ClientError in _get_symbol_trading_rules (Attempt {attempt + 1}/{retries}) for {self.symbol}: "
                                  f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.headers}")
                if attempt < retries - 1: time.sleep(delay)
                else: return False
            except requests.exceptions.RequestException as e: 
                self.logger.error(f"Network error in _get_symbol_trading_rules (Attempt {attempt + 1}/{retries}) for {self.symbol}: {e}")
                if attempt < retries - 1: time.sleep(delay * (attempt + 1))
                else: return False
            except Exception as e:
                self.logger.error(f"Unexpected error in _get_symbol_trading_rules (Attempt {attempt + 1}/{retries}) for {self.symbol}: {e}", exc_info=True)
                return False 
        return False

    def _get_and_store_leverage_data(self, retries: int = 3, delay: int = 5):
        for attempt in range(retries):
            try:
                self.logger.info(f"Requesting leverage brackets (Attempt {attempt + 1}/{retries}) for {self.symbol}...")
                data = self.client.leverage_brackets(symbol=self.symbol)

                if data and isinstance(data, list) and len(data) > 0:
                    symbol_data = data[0]
                    self.leverage_brackets_for_symbol = symbol_data.get('brackets', [])
                    if self.leverage_brackets_for_symbol:
                        self.logger.info(f"Leverage bracket data for {self.symbol} stored. Found {len(self.leverage_brackets_for_symbol)} brackets.")
                    else:
                        self.logger.warning(f"Data received for {self.symbol}, but no 'brackets' found. Using default leverage brackets.")
                        self.leverage_brackets_for_symbol = [{'bracket': 1, 'initialLeverage': 20, 'notionalCap': 1000000000, 'notionalFloor': 0}]
                    return True
                elif data is None:
                    self.logger.warning(f"API response for leverage brackets for {self.symbol} was None (Attempt {attempt + 1}/{retries}).")
                    if attempt < retries - 1:
                        self.logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"All {retries} attempts failed to get a valid API response (got None) for leverage brackets for {self.symbol}.")
                else:
                    self.logger.warning(f"Received invalid or empty data format for leverage brackets for {self.symbol} (Attempt {attempt + 1}/{retries}). Data: {data}")
                    if attempt < retries - 1:
                        self.logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
            except ClientError as e:
                self.logger.error(f"API ClientError fetching leverage brackets (Attempt {attempt + 1}/{retries}) for {self.symbol}: "
                                  f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.headers}")
                if attempt < retries - 1:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error fetching leverage brackets (Attempt {attempt + 1}/{retries}) for {self.symbol}: {e}")
                if attempt < retries - 1:
                    self.logger.info(f"Retrying in {delay * (attempt + 1)} seconds...")
                    time.sleep(delay * (attempt + 1))
            except Exception as e:
                self.logger.error(f"Unexpected error fetching leverage brackets (Attempt {attempt + 1}/{retries}) for {self.symbol}: {e}", exc_info=True)
                break 
            
        self.logger.warning(f"Failed to fetch and store valid leverage brackets for {self.symbol} after all retries. Using default values.")
        self.leverage_brackets_for_symbol = [{'bracket': 1, 'initialLeverage': 20, 'notionalCap': 1000000000, 'notionalFloor': 0}]
        return False

    
    def _set_isolated_margin(self):
        try:
            self.logger.info(f"Checking margin type for {self.symbol}...")
            # Get position risk (shows current margin type)
            positions = self.client.get_position_risk(symbol=self.symbol)
            
            if positions:
                # Use first position as margin type is shared for the pair
                current_margin_type = positions[0].get('marginType', '').lower()
                if current_margin_type == 'isolated':
                    self.logger.info(f"Margin type is already ISOLATED for {self.symbol}.")
                    return True

            self.logger.info(f"Attempting to set margin type for {self.symbol} to ISOLATED.")
            self.client.change_margin_type(symbol=self.symbol, marginType='ISOLATED')
            self.logger.info(f"Margin type for {self.symbol} successfully set to ISOLATED.")
            return True

        except ClientError as e:
            if e.error_code == -4046: # "No need to change margin type"
                self.logger.info(f"Margin type for {self.symbol} is already ISOLATED.")
                return True
            self.logger.error(f"API Error setting margin: {e.error_message}")
            return False
        except Exception as e:
            self.logger.error(f"Error setting margin: {e}", exc_info=True)
            return False
    
    def _get_account_balance_usdt(self, asset: str = 'USDT', retries: int = 3, delay: int = 5):
        for i in range(retries):
            try:
                balances = self.client.balance(recvWindow=6000)
                for acc_balance in balances:
                    if acc_balance['asset'] == asset:
                        return float(acc_balance['balance'])
                self.logger.warning(f"Asset {asset} not found in futures account balance ({self.symbol}, attempt {i+1}/{retries}).")
                if i < retries - 1: time.sleep(delay)
                else: return None 
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e_net:
                log_level = logging.INFO if i < retries - 1 else logging.WARNING
                self.logger.log(log_level, f"Network error getting balance ({self.symbol}, attempt {i+1}/{retries}): {type(e_net).__name__} - {e_net}")
                if i < retries - 1: self.logger.debug(f"Retrying in {delay} seconds...")
                else: self.logger.error(f"All {retries} attempts failed to get balance for {self.symbol} (Network Error).")
                if i < retries - 1: time.sleep(delay)
                else: return None
            except ClientError as e_client:
                log_level = logging.WARNING if i < retries - 1 else logging.ERROR
                self.logger.log(log_level, f"Binance API Error getting balance ({self.symbol}, attempt {i+1}/{retries}): Status {e_client.status_code}, Code {e_client.error_code}, Msg: {e_client.error_message}")
                if i < retries - 1: self.logger.info(f"Retrying in {delay} seconds...")
                else: self.logger.error(f"All {retries} attempts failed to get balance for {self.symbol} (API Error).")
                if i < retries - 1: time.sleep(delay)
                else: return None
            except Exception as e_other:
                self.logger.error(f"Unexpected error getting balance ({self.symbol}, attempt {i+1}/{retries}): {e_other}", exc_info=True)
                if i < retries - 1: time.sleep(delay)
                else: return None
        return None

    def _check_capital_and_stop_if_needed(self, reason: str):
        """Checks if virtual capital has reached the stop threshold."""
        # Calculate stop threshold: initial capital * (1 - drawdown percent)
        stop_threshold = self.bot_starting_virtual_capital * (Decimal('1.0') - self.bot_virtual_capital_stop_loss_percent)

        if self.virtual_bot_capital_usdt <= stop_threshold:
            self.bot_stopped_due_to_total_sl = True # Set main stop flag
            self.logger.critical(
                f"!!! BOT STOP LOSS !!! Capital {self.virtual_bot_capital_usdt:.4f} USDT reached or fell below threshold {stop_threshold:.4f} USDT."
            )
            self.logger.critical(f"Bot for {self.symbol} stopped. Reason: {reason}")
            return True
        return False

    def _get_open_positions(self, symbol_to_check: str = None, retries: int = 3, delay: int = 5):
        params_query = {}
        current_symbol_query = symbol_to_check if symbol_to_check else self.symbol
        log_msg_symbol = current_symbol_query if current_symbol_query else "all symbols"
        
        if current_symbol_query:
             params_query['symbol'] = current_symbol_query

        for attempt in range(retries):
            try:
                self.logger.debug(f"Requesting position risk (Attempt {attempt+1}/{retries}) for {log_msg_symbol}...")
                positions = self.client.get_position_risk(**params_query, recvWindow=6000)
                
                if current_symbol_query: 
                    for position in positions: 
                        if position['symbol'] == current_symbol_query and float(position.get('positionAmt', 0)) != 0:
                            self.logger.info(f"Active position found for {current_symbol_query}: Amount {position['positionAmt']}, Entry {position['entryPrice']}")
                            return [position] 
                    self.logger.info(f"No active position found for {current_symbol_query} (Attempt {attempt+1}/{retries}).")
                    return [] 
                else: 
                    active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
                    if active_positions:
                        self.logger.info(f"Found {len(active_positions)} active position(s).")
                    else:
                        self.logger.info(f"No active positions found on the account (Attempt {attempt+1}/{retries}).")
                    return active_positions

            except ClientError as e:
                # --- CRASH FIX ---
                self.logger.error(f"API ClientError getting position risk (Attempt {attempt + 1}/{retries}) for {log_msg_symbol}: "
                                  f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}")
                
                # If Invalid Key/IP error (-2015), stop retrying
                if e.error_code == -2015:
                    self.logger.critical("⛔ FATAL API ERROR (-2015): Invalid Key, IP, or Permissions. Stop retrying.")
                    return []

                if attempt < retries - 1:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error getting position risk (Attempt {attempt + 1}/{retries}) for {log_msg_symbol}: {e}")
                if attempt < retries - 1:
                    self.logger.info(f"Retrying in {delay * (attempt+1)} seconds...")
                    time.sleep(delay * (attempt+1))

            except Exception as e:
                self.logger.error(f"Unexpected error getting position risk (Attempt {attempt + 1}/{retries}) for {log_msg_symbol}: {e}", exc_info=True)
                return [] 
        
        self.logger.error(f"All {retries} attempts failed for getting position risk for {log_msg_symbol}.")
        return []

    def _get_klines_df_rest(self, klines_interval: str, klines_limit: int):
        try:
            self.logger.info(f"Requesting historical Klines for {self.symbol} {klines_interval}, limit {klines_limit}...")
            klines_data = self.client.klines(symbol=self.symbol, interval=klines_interval, limit=klines_limit)
            if not klines_data:
                self.logger.warning(f"No klines data received for {self.symbol} {klines_interval}.")
                return pd.DataFrame()

            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
            # --- SYNC WITH get_all_historical_futures_data ---
            df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # --- END OF BLOCK ---

            df.set_index('timestamp', inplace=True)
            self.logger.info(f"Successfully loaded {len(df)} klines for {self.symbol} {klines_interval}.")
            return df
        except ClientError as e:
            self.logger.error(f"API error fetching klines for {self.symbol} {klines_interval}: {e.error_message}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching klines for {self.symbol} {klines_interval}: {e}", exc_info=True)
            return pd.DataFrame()
            
    def _manage_user_data_stream(self, current_listen_key: str = None, max_retries: int = 3, retry_delay: int = 10):
        key_to_renew = current_listen_key if current_listen_key else self.listen_key
        if key_to_renew:
            try:
                self.logger.info(f"Attempting to renew existing listen key: {key_to_renew[:5]}... for {self.symbol}")
                self.client.renew_listen_key(listenKey=key_to_renew)
                self.last_listen_key_renewal_time = time.time()
                self.logger.info(f"User Data Stream listen key renewed successfully: {key_to_renew[:5]}... for {self.symbol}")
                return key_to_renew
            except ClientError as e_renew:
                self.logger.warning(f"Failed to renew listen key {key_to_renew[:5]}... for {self.symbol} (Code: {e_renew.error_code}, Msg: {e_renew.error_message}). Attempting to get a new one.")
            except Exception as e_renew_general:
                self.logger.error(f"Unexpected error renewing listen key {key_to_renew[:5]}... for {self.symbol}: {e_renew_general}", exc_info=True)

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting to get a new listen key for User Data Stream (Attempt {attempt + 1}/{max_retries}) for {self.symbol}...")
                key_response = self.client.new_listen_key()
                new_key = key_response.get('listenKey')
                if new_key:
                    self.logger.info(f"Successfully obtained new listenKey: {new_key[:5]}... for {self.symbol}")
                    self.last_listen_key_renewal_time = time.time() 
                    return new_key 
                else:
                    self.logger.error(f"Failed to get a new listenKey from API response (Attempt {attempt + 1}/{max_retries}) for {self.symbol}. Response: {key_response}")
            except ClientError as e_new:
                self.logger.error(f"API ClientError getting new listen key (Attempt {attempt + 1}/{max_retries}) for {self.symbol}: "
                                     f"Status {e_new.status_code}, Code {e_new.error_code}, Msg: {e_new.error_message}")
            except requests.exceptions.RequestException as e_req: 
                self.logger.error(f"Network error getting new listen key (Attempt {attempt + 1}/{max_retries}) for {self.symbol}: {e_req}")
            except Exception as e_new_general:
                self.logger.error(f"Unexpected error getting new listen key (Attempt {attempt + 1}/{max_retries}) for {self.symbol}: {e_new_general}", exc_info=True)
            
            if attempt < max_retries - 1:
                self.logger.info(f"Retrying to get new listen key in {retry_delay} seconds for {self.symbol}...")
                time.sleep(retry_delay)

        self.logger.critical(f"Failed to obtain/renew listen key after all attempts for {self.symbol}. User Data Stream functionality will be affected.")
        return None
    
    def _on_websocket_open(self, ws_client_instance):
        self.logger.info(f"WebSocket connection successfully opened for {self.symbol}.")
        self.reconnect_websocket_flag = False

    def _on_websocket_close(self, ws_client_instance, close_status_code=None, close_msg=None):
        self.logger.critical(f"!!!!!!!!!! ON_WEBSOCKET_CLOSE CALLED for {self.symbol} !!!!!!!!!!")
        if self.bot_stopped_due_to_total_sl or self.should_stop_bot_flag_instance:
            self.logger.info(f"WebSocket closed for {self.symbol}, but bot is already flagged to stop. No reconnect action.")
            return

        if close_status_code is not None or close_msg is not None:
            self.logger.warning(f"WebSocket connection for {self.symbol} closed: Status Code: {close_status_code}, Message: '{close_msg}'.")
        else:
            self.logger.warning(f"WebSocket connection for {self.symbol} closed (no specific status/msg provided).")

        if not self.reconnect_websocket_flag: 
            self.logger.info(f"Flagging for WebSocket re-initialization by the main loop for {self.symbol}.")
            self.reconnect_websocket_flag = True

    def _on_websocket_error(self, ws_client_instance, error):
        self.logger.error(f"WebSocket error reported for {self.symbol}: {error}")
        self.logger.critical(f"!!!!!!!!!! ON_WEBSOCKET_ERROR CALLED for {self.symbol} !!!!!!!!!! Error: {error}")

        error_str = str(error).lower()
        connection_lost_errors = [
            "connection to remote host was lost", "connection closed", 
            "connection aborted", "websocket connection close"
        ]
        is_connection_loss = any(phrase in error_str for phrase in connection_lost_errors)

        if is_connection_loss:
            if not self.bot_stopped_due_to_total_sl and \
               not self.should_stop_bot_flag_instance and \
               not self.reconnect_websocket_flag:
                self.logger.warning(f"WebSocket error for {self.symbol} indicates a connection loss. Flagging for re-initialization.")
                self.reconnect_websocket_flag = True
    
    def _reset_position_state(self):
        """Resets internal position state of the bot."""
        self.logger.info(f"Resetting internal position state for {self.symbol}.")
        
        self.current_position_on_exchange = None
        self.current_position_entry_price = 0.0
        self.current_position_quantity = 0.0
        
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        
        # --- FIX: RESET BE FLAG FOR NEXT TRADE ---
        self.is_be_activated = False  # <--- ОСЬ ЦЬОГО НЕ ВИСТАЧАЛО!
        
        self.is_processing_position_close = False 
        self._cancel_trap_orders()

        # Reset limit entry order details if it was active
        if self.active_limit_entry_order_details and self.active_limit_entry_order_details.get("orderId"):
            self._reset_active_limit_entry_order_details()

    def _get_current_position_info(self):
        """Gets fresh position info for the current symbol from the exchange."""
        try:
            self.logger.debug(f"Requesting fresh position info for {self.symbol}...")
            # Use get_position_risk as it provides details for specific symbol
            positions = self.client.get_position_risk(symbol=self.symbol)
            if positions:
                # Return the first (and only) dictionary from the list
                return positions[0] 
        except Exception as e:
            self.logger.error(f"Error getting current position info for {self.symbol}: {e}")
        return None    
    
    def _start_websocket_listener(self):
        """
        Starts WebSocket client (Extracted method for Lab/Test).
        """
        self.logger.info("🔌 Starting WebSocket Client...")
        print("DEBUG: Starting WebSocket...") # Direct output
        
        # 1. Get Listen Key
        if not self.listen_key:
            self.listen_key = self._manage_user_data_stream()
            if not self.listen_key:
                self.logger.error("Failed to generate listen key.")
                return

        # 2. Create client (using self.ws_client)
        self.ws_client = UMFuturesWebsocketClient(
            on_message=self._on_websocket_message,
            on_close=self._on_websocket_close,
            on_error=self._on_websocket_error,
            on_open=self._on_websocket_open,
            is_combined=True
        )
        
        # 3. Subscribe to User Data (Mandatory!)
        self.ws_client.user_data(
            listen_key=self.listen_key,
            id=0,
            callback=self._on_websocket_message
        )
        
        # 4. Subscribe to Klines (to keep connection alive)
        self.ws_client.kline(
            symbol=self.symbol, 
            id=1, 
            interval=self.interval, 
            callback=self._on_websocket_message
        )
        
        self.logger.info("✅ WebSocket streams subscribed.")
        print("DEBUG: WebSocket subscribed.")

    def _continue_sequence_immediately(self, last_exit_price):
        """
        CHAIN REACTION: Immediate entry after close.
        FIXED: Smart Reversal logic (Win -> Flip, Loss -> Keep).
        """
        # Even if steps is 0, we allow this last entry, 
        # but counter will be checked before decrement.
        if self.sequence_steps_remaining < 0: return

        self.logger.info(f"🔗 CHAIN REACTION: Trade closed @ {last_exit_price}. Snap & Go!")

        # 1. Determine where price went (Trend Direction)
        # Compare exit price with previous remembered level
        step_base = self.last_analyzed_step_level
        if step_base is None or step_base == 0:
            step_base = float(last_exit_price)

        # Actual market direction
        market_direction = 1 if float(last_exit_price) > step_base else -1
        
        # 2. Side selection logic
        if self.active_sequence_mode == 'FORCE_ZEROS':
            # REVERSAL Strategy:
            # We always go AGAINST the market.
            # If market went UP (market_direction = 1) -> We want SELL (target = -1)
            # If market went DOWN (market_direction = -1) -> We want BUY (target = 1)
            self.current_session_direction = -1 * market_direction
            
            dir_str = "DOWN (Short)" if self.current_session_direction == -1 else "UP (Long)"
            self.logger.info(f"🔄 SMART REVERSAL: Market went {market_direction}. Next Target: {dir_str}")

        elif self.active_sequence_mode == 'FORCE_ONES':
            # TREND Strategy (Breakout):
            # We follow the market.
            self.current_session_direction = market_direction
            
        # 3. Get parameters
        x_val = self.current_session_x
        direction = self.current_session_direction
        
        # Emergency X
        if x_val <= 0: x_val = float(last_exit_price) * 0.001

        # 4. Snap to Grid
        snapped_price = round(float(last_exit_price) / x_val) * x_val
        
        target_price = snapped_price
        trap_side = None
        side = None

        # 5. Determine order
        if direction == 1: # Target UP
            side = 'BUY'
            trap_side = 'UP' 
            target_price = snapped_price 

        elif direction == -1: # Target DOWN
            side = 'SELL'
            trap_side = 'DOWN'
            target_price = snapped_price

        if side and trap_side:
            self.logger.info(f"🔗 CHAIN ENTRY: {side} @ {target_price:.4f} (Snapped from {last_exit_price}, X={x_val})")
            
            self._update_or_create_trap_order(trap_side, side, target_price, x_val)
            
            if self.sequence_steps_remaining > 0:
                self.sequence_steps_remaining -= 1
            
            if self.sequence_steps_remaining == 0:
                self.logger.info("🏁 SEQUENCE FINISHED (Counter hit 0). Last order placed.")
                # Do NOT reset mode here, so Guard Clause still works while order is pending

    def _on_websocket_message(self, ws, message):
        """
        WebSocket message processing.
        SYNC IMMUNITY ADDED + Iron Grip Fix + No Breakeven.
        """
        # 1. GLOBAL STOP CHECK
        if self.bot_stopped_due_to_total_sl or self.should_stop_bot_flag_instance:
            try:
                if isinstance(message, str): msg_data = json.loads(message)
                else: msg_data = message
                if 'data' in msg_data: msg_data = msg_data['data']
                if msg_data.get('e') == 'ORDER_TRADE_UPDATE':
                    o = msg_data.get('o', {})
                    self.logger.info(f"🛑 IGNORED WS UPDATE during SHUTDOWN: ID {o.get('i')} Status {o.get('X')}")
            except: pass
            return

        try:
            if isinstance(message, str): msg_data = json.loads(message)
            else: msg_data = message
            
            if 'data' in msg_data and isinstance(msg_data['data'], dict):
                msg_data = msg_data['data']

            event_type = msg_data.get('e')

            # --- 1. KLINE (PRICE UPDATES) ---
            if event_type == 'kline':
                k_data = msg_data['k']
                symbol_ws = k_data['s']
                interval_ws = k_data['i']
                is_candle_closed = k_data['x']
                kline_open_time_ws = pd.to_datetime(k_data['t'], unit='ms')

                if symbol_ws != self.symbol: return

                current_time = time.time()
                # Heartbeat
                if current_time - self.last_heartbeat_log_time > 300:
                    uptime = int(current_time - self.start_time)
                    uptime_str = time.strftime('%H:%M:%S', time.gmtime(uptime))
                    pos_str = f"{self.current_position_on_exchange}" if self.current_position_on_exchange else "None"
                    self.logger.info(f"Heartbeat {self.symbol}. Uptime: {uptime_str}. Pos: {pos_str}. Cap: {self.virtual_bot_capital_usdt:.4f}.")
                    self.last_heartbeat_log_time = current_time

                if symbol_ws == self.symbol:
                    if interval_ws == self.interval: 
                        self.klines_df_main = self._update_klines_with_new_data(self.klines_df_main, msg_data, self.limit_main_tf)
                        self.last_processed_kline_open_time = kline_open_time_ws

                        # --- ENTRY TRIGGER ---
                        # (Перевірка БУ видалена, залишився тільки вхід)
                        if not self.current_position_on_exchange:
                            if not hasattr(self, 'last_ai_check_time'): self.last_ai_check_time = 0
                            if (current_time - self.last_ai_check_time) > 15:
                                self.last_ai_check_time = current_time
                                if not self.klines_df_main.empty and not self.is_processing_position_close:
                                    self._check_and_execute_trade_logic(self.klines_df_main)
                                    
                    elif interval_ws == self.selected_timeframe_higher and is_candle_closed:
                        self.klines_df_higher_tf = self._update_klines_with_new_data(self.klines_df_higher_tf, msg_data, self.limit_higher_tf)

            # --- 2. ORDER UPDATE (ACCOUNTING CORE) ---
            elif event_type == 'ORDER_TRADE_UPDATE':
                order_data = msg_data.get('o', {})
                if order_data.get('s') != self.symbol: return

                order_id_ws = str(order_data.get('i'))
                order_status_ws = order_data.get('X')
                order_type_ws = order_data.get('ot') 
                order_side_ws = order_data.get('S') 
                rp_pnl = Decimal(order_data.get('rp', '0'))

                if order_status_ws in ['FILLED', 'CANCELED', 'NEW', 'REJECTED']:
                     self.logger.info(f"ORDER UPDATE: ID {order_id_ws}, Status {order_status_ws}, Type {order_type_ws}, PnL {rp_pnl}")

                # --- 2.1 ENTRY FILL ---
                is_entry_fill = False
                if order_status_ws == 'FILLED':
                    is_reduce = order_data.get('R', False)
                    if not is_reduce and rp_pnl == 0:
                        is_entry_fill = True
                
                if is_entry_fill:
                    if self.should_stop_bot_flag_instance: return
                    self.logger.info(f"🪤 TRAP ACTIVATED (Iron Grip)! Order {order_id_ws} filled.")
                    
                    try:
                        pos_side = 'LONG' if order_side_ws == 'BUY' else 'SHORT'
                        qty_filled = float(order_data.get('z', '0.0'))
                        price_filled = float(order_data.get('ap', '0.0'))
                        if price_filled == 0: price_filled = float(order_data.get('L', '0.0'))
                        
                        if order_data.get('N') == 'USDT':
                            commission = Decimal(order_data.get('n', '0'))
                            self.virtual_bot_capital_usdt -= commission
                            self.logger.info(f"📉 Entry Fee Deducted: {commission} USDT. Cap: {self.virtual_bot_capital_usdt:.4f}")

                        # Context Recovery / Strategy Params (FIXED KEYERROR)
                        pending_ctx = getattr(self, 'pending_ai_trade_details', None)
                        
                        # Перевіряємо наявність ключів, щоб не зловити KeyError
                        if pending_ctx and 'sl_pct' in pending_ctx:
                            self.logger.info(f"🧠 AI Context Restored!")
                            sl_pct = Decimal(str(pending_ctx['sl_pct']))
                            tp_pct = Decimal(str(pending_ctx['tp_pct']))
                            self.pending_ai_trade_details = None
                        else:
                            self.logger.warning(f"⚠️ Context Partial or Lost. Using Default Strategy Params.")
                            sl_pct = Decimal(str(self.ai_stop_loss_pct))
                            tp_pct = Decimal(str(self.ai_take_profit_pct))

                        if price_filled > 0:
                            p_dec = Decimal(str(price_filled))
                            if pos_side == 'LONG':
                                target_sl = p_dec * (Decimal('1') - sl_pct)
                                target_tp = p_dec * (Decimal('1') + tp_pct)
                            else:
                                target_sl = p_dec * (Decimal('1') + sl_pct)
                                target_tp = p_dec * (Decimal('1') - tp_pct)
                        
                        self.current_position_on_exchange = pos_side
                        self.current_position_quantity = qty_filled
                        self.current_position_entry_price = Decimal(str(price_filled))
                        
                        self.logger.info(f"✅ POSITION OPENED: {pos_side} {qty_filled} @ {price_filled}")
                        
                        self._place_sl_tp_orders_after_entry(
                            position_side=self.current_position_on_exchange,
                            position_qty=self.current_position_quantity,
                            entry_price=self.current_position_entry_price,
                            sl_price_from_signal=target_sl,
                            tp_price_from_signal=target_tp,
                            price_precision=self.symbol_trading_rules.get('pricePrecision', 2),
                            tick_size=self.symbol_trading_rules.get('tickSize', '0.01')
                        )
                            
                    except Exception as e_grip:
                        self.logger.critical(f"❌ CRITICAL ERROR IN IRON GRIP LOGIC: {e_grip}", exc_info=True)

                # --- 2.2 EXIT FILL ---
                elif order_status_ws == 'FILLED' and not is_entry_fill:
                    if self.current_position_on_exchange:
                        if not self.is_processing_position_close:
                            self.is_processing_position_close = True
                            
                            # --- 🔥 RECORD CANDLE FOR COOLDOWN 🔥 ---
                            self.last_trade_candle_time = self.last_processed_kline_open_time
                            self.logger.info(f"🥶 COOLDOWN ACTIVATED: Blocked entries for candle {self.last_trade_candle_time}")
                            
                            exit_label = "LIMIT/MARKET"
                            if 'TAKE_PROFIT' in order_type_ws: exit_label = 'TAKE PROFIT'
                            elif 'STOP' in order_type_ws: exit_label = 'STOP LOSS'
                            elif 'LIQUIDATION' in order_type_ws: exit_label = 'LIQUIDATION'
                            
                            exit_fill_price = float(order_data.get('ap', '0.0'))
                            if exit_fill_price == 0: exit_fill_price = float(order_data.get('L', '0.0'))
                            
                            filled_qty = float(order_data.get('z', '0.0'))
                            entry_calc = float(self.current_position_entry_price)
                            
                            real_pnl = 0.0
                            if self.current_position_on_exchange == 'LONG':
                                real_pnl = (exit_fill_price - entry_calc) * filled_qty
                            else: 
                                real_pnl = (entry_calc - exit_fill_price) * filled_qty
                                
                            comm = float(order_data.get('n', '0'))
                            net_pnl = real_pnl - comm

                            self.virtual_bot_capital_usdt += Decimal(str(net_pnl))
                            
                            delta = self.virtual_bot_capital_usdt - self.bot_starting_virtual_capital
                            delta_sign = "+" if delta >= 0 else ""
                            pnl_sign = "+" if net_pnl > 0 else ""
                            result_type = "WIN" if net_pnl > 0 else "LOSS"

                            report = f"\n {result_type} {self.symbol} CLOSED\n"
                            report += f" Type: {exit_label}\n"
                            report += f" PnL: {pnl_sign}{net_pnl:.4f} USDT (Real: {real_pnl:.4f} - Comm: {comm:.4f})\n"
                            report += f" Balance: {self.virtual_bot_capital_usdt:.4f} USDT\n"
                            report += f" Delta: {delta_sign}{delta:.4f} USDT"
                            
                            self.logger.info(report)

                            self._check_capital_and_stop_if_needed("End of Trade")
                            self._cancel_all_orders_for_symbol()
                            self._reset_position_state()

                # --- 2.3 CANCELLATION ---
                elif order_status_ws in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    if self.trap_order_up and str(order_id_ws) == str(self.trap_order_up.get('orderId')):
                        self.trap_order_up = None
                    elif self.trap_order_down and str(order_id_ws) == str(self.trap_order_down.get('orderId')):
                        self.trap_order_down = None

            # --- 3. SYNC ---
            elif event_type == 'ACCOUNT_UPDATE':
                # --- 🔥 SYNC IMMUNITY CHECK 🔥 ---
                # Якщо ми недавно (30с) ставили ордер або БУ - ігноруємо цей апдейт, 
                # щоб не скинути налаштування через лаг WebSocket.
                if time.time() < getattr(self, 'ignore_sync_timestamp', 0):
                    # self.logger.info("🛡️ SYNC IGNORED (Immunity Active)")
                    return
                # ---------------------------------

                acc_data = msg_data.get('a', {})
                for pos in acc_data.get('P', []):
                    if pos.get('s') == self.symbol:
                        amt = float(pos.get('pa', '0'))
                        if self.should_stop_bot_flag_instance: return

                        if amt != 0 and self.current_position_on_exchange is None:
                            if self.trap_order_up is not None or self.trap_order_down is not None:
                                return

                            side = 'LONG' if amt > 0 else 'SHORT'
                            self.logger.warning(f"⚠️ SYNC TRIGGERED: Found {side} {amt} position via ACCOUNT_UPDATE.")
                            
                            self.current_position_on_exchange = side
                            self.current_position_quantity = abs(amt)
                            self.current_position_entry_price = Decimal(pos.get('ep', '0'))
                            
                            self.logger.warning(f"⚠️ SYNC: Generating EMERGENCY SL/TP.")
                            
                            entry_p = self.current_position_entry_price
                            sl_pct = Decimal(str(self.ai_stop_loss_pct))
                            tp_pct = Decimal(str(self.ai_take_profit_pct))
                            
                            if side == 'LONG':
                                sl = entry_p * (Decimal('1') - sl_pct)
                                tp = entry_p * (Decimal('1') + tp_pct)
                            else:
                                sl = entry_p * (Decimal('1') + sl_pct)
                                tp = entry_p * (Decimal('1') - tp_pct)
                            
                            self._place_sl_tp_orders_after_entry(
                                position_side=self.current_position_on_exchange,
                                position_qty=self.current_position_quantity,
                                entry_price=self.current_position_entry_price,
                                sl_price_from_signal=sl,
                                tp_price_from_signal=tp,
                                price_precision=self.symbol_trading_rules.get('pricePrecision', 2),
                                tick_size=self.symbol_trading_rules.get('tickSize', '0.01')
                            )

        except Exception as e:
            self.logger.error(f"WS Error {self.symbol}: {e}", exc_info=True)

    def _update_klines_with_new_data(self, current_df: pd.DataFrame, new_kline_data: dict, max_rows: int):
        try:
            k_data = new_kline_data['k']
            kline_open_time = pd.to_datetime(k_data['t'], unit='ms')

            new_data_values = {
                'open': float(k_data['o']), 'high': float(k_data['h']),
                'low': float(k_data['l']), 'close': float(k_data['c']),
                'volume': float(k_data['v'])
            }
            if kline_open_time in current_df.index:
                for col, value in new_data_values.items():
                    current_df.loc[kline_open_time, col] = value
            else:
                new_row = pd.DataFrame([new_data_values], index=[kline_open_time])
                current_df = pd.concat([current_df, new_row])

            if len(current_df) > max_rows:
                current_df = current_df.iloc[-max_rows:]

            current_df.sort_index(inplace=True)
            return current_df
        except Exception as e:
            self.logger.error(f"Error updating klines_df for {self.symbol}: {e}, data: {new_kline_data}", exc_info=True)
            return current_df

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            self.logger.info(f"ML model successfully loaded from file: {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None


    def _calculate_indicators(self, df_main: pd.DataFrame, df_higher: pd.DataFrame):
        """
        Calculating indicators using High/Low logic for steps.
        """
        if df_main.empty or df_higher.empty:
            self.logger.error("!!! INDICATORS: Initial data empty.")
            return pd.DataFrame()

        df = df_main.copy()
        df_tf = df_higher.copy()

        # --- ATR from higher TF ---
        high_tf, low_tf, close_tf_prev = df_tf['high'], df_tf['low'], df_tf['close'].shift(1)
        tr_tf = pd.concat([high_tf - low_tf, abs(high_tf - close_tf_prev), abs(low_tf - close_tf_prev)], axis=1).max(axis=1, skipna=False)
        df_tf['atr'] = tr_tf.rolling(window=self.atr_period).mean()
        df_tf['atr'].ffill(inplace=True)
        df_tf_sec = df_tf[['close', 'atr']].copy()
        df_tf_sec.rename(columns={'close': 'close_tf', 'atr': 'atr_tf'}, inplace=True)
        df = pd.merge_asof(df.sort_index(), df_tf_sec.sort_index(), left_index=True, right_index=True, direction='backward')

        df.dropna(subset=['atr_tf', 'close_tf'], inplace=True)
        if df.empty: return pd.DataFrame()

        # --- Calculating X ---
        percent = (df['atr_tf'] / df['close_tf']) * 100
        idx = percent * df['close'] / 100
        idx.replace([np.inf, -np.inf], np.nan, inplace=True)
        idx.ffill(inplace=True)
        idx.replace(0, 1e-10, inplace=True)

        valid_idx_mask = (idx > 0) & pd.notna(idx)
        magnitude = pd.Series(np.nan, index=df.index)
        if valid_idx_mask.any():
            magnitude.loc[valid_idx_mask] = 10 ** np.floor(np.log10(idx[valid_idx_mask]))
        magnitude.ffill(inplace=True)
        magnitude.fillna(1e-10, inplace=True)

        safe_magnitude = magnitude.replace(0, 1e-10)
        residual = (idx / safe_magnitude).replace([np.inf, -np.inf], np.nan).ffill()
        df['smooth_residual'] = residual.ewm(alpha=(1/self.residual_smooth), adjust=False).mean().ffill()

        conditions = [df['smooth_residual'] < 1.5, (df['smooth_residual'] >= 1.5) & (residual < 3.5), (df['smooth_residual'] >= 1.5) & (residual >= 3.5) & (residual < 7.5)]
        choices = [magnitude * 1, magnitude * 2, magnitude * 5]
        
        base_x = np.select(conditions, choices, default=magnitude * 10)
        current_coeff = getattr(self, 'step_coeff', 0.5)
        df['x'] = base_x * current_coeff
        
        df['x'].replace(0, np.nan, inplace=True)
        df['x'].ffill(inplace=True)
        df.dropna(subset=['x', 'close', 'high', 'low'], inplace=True)
        
        if df.empty: return pd.DataFrame()

        # --- !!! CALLING NUMBA WITH 6 ARGUMENTS !!! ---
        high_np = df['high'].to_numpy(dtype=np.float64)
        low_np = df['low'].to_numpy(dtype=np.float64)
        close_np = df['close'].to_numpy(dtype=np.float64)
        x_np = df['x'].to_numpy(dtype=np.float64)

        # # === OPTION 1: LOGIC (HL Only + Dynamic X) ===
        # Define step grid directly here
        step_grid_np = np.array([
            0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 
            25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0
        ], dtype=np.float64)

        start_price = close_np[0] if len(close_np) > 0 else 0.0

        # Call with all arguments
        df['step_level'] = calculate_step_level_numba_hl(
            high_np, 
            low_np, 
            close_np, 
            x_np, 
            step_grid_np, 
            start_price
        )
        # # === OPTION 2: NEW LOGIC (Close Only + Dynamic X) ===
        # # We need the rounded_close array for the new function. 
        # # Calculate it vectorially right here:
        # rounded_close_np = np.round(close_np / x_np) * x_np
        
        # # Call the new function (only 3 arguments required!)
        # df['step_level'] = calculate_step_level_numba(
        #     close_np, 
        #     rounded_close_np, 
        #     x_np
        # )

        df['step_level'].ffill(inplace=True)
        df.dropna(subset=['step_level'], inplace=True)

        # --- Other indicators ---
        df['prev_step_level'] = df['step_level'].shift(1)
        df['step_count_in_trend'] = calculate_step_index(df['step_level'].to_numpy())
        
        if 'atr_tf' in df.columns:
            df['safe_atr'] = df['atr_tf'].ffill().replace(0, np.nan).ffill().fillna(1e-9)
        else:
            df['safe_atr'] = 1e-9
            
        if 'x' in df.columns:
            df['safe_x'] = df['x'].replace(0, np.nan).ffill().fillna(1e-9)
        else:
            df['safe_x'] = 1e-9

        df['step_index_from_x'] = (df['step_level'] / df['safe_x']).round().fillna(0).astype(int)
        df['pivot_step_index'] = _calculate_pivot_step_index_price_based(
            df['step_level'].to_numpy(), df['safe_x'].to_numpy(), reset_threshold=3 
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df
    
    def _place_blind_trade(self, df, decision, current_direction):
        """
        Blind entry in sequence.
        Strict price calculation from STEP LEVEL +/- X (no magic numbers).
        """
        last_row = df.iloc[-1]
        current_step = float(last_row.get('step_level', 0))
        raw_x = float(last_row.get('x', 0))
        
        # Zero protection
        if current_step == 0: return
        # If X is 0 for some reason, take a tiny percentage to avoid division error
        x_val = raw_x if raw_x > 0 else (current_step * 0.001)

        # Determine "Front" - where the next level is
        # If trend UP -> Next level = Step + X
        # If trend DOWN -> Next level = Step - X
        
        target_price = 0.0
        trap_side = None
        side = None
        
        if current_direction == 1: # Trend UP
            next_level = current_step + x_val
            
            if decision == 1: # Force ONES (Breakout UP)
                side = 'BUY'
                trap_side = 'UP' # Buy Stop
                target_price = next_level 
            
            elif decision == 0: # Force ZEROS (Reversal DOWN)
                side = 'SELL'
                trap_side = 'UP' # Sell Limit (at resistance)
                target_price = next_level

        elif current_direction == -1: # Trend DOWN
            next_level = current_step - x_val
            
            if decision == 1: # Force ONES (Breakout DOWN)
                side = 'SELL'
                trap_side = 'DOWN' # Sell Stop
                target_price = next_level 
            
            elif decision == 0: # Force ZEROS (Reversal UP)
                side = 'BUY'
                trap_side = 'DOWN' # Buy Limit (at support)
                target_price = next_level
        
        if side and trap_side:
            self.logger.info(f"🎲 BLIND TRADE (Mode {decision}): Dir {current_direction}. Placing {side} @ {target_price:.5f} (Step={current_step:.5f}, X={x_val:.5f})")
            self._update_or_create_trap_order(trap_side, side, target_price, x_val)

    def _manual_position_sync(self):
        """
        PARANOID SYNC.
        Forcefully queries API for position state.
        If a 'ghost' position (not in memory) is found:
        1. Update memory.
        2. Place EMERGENCY STOPS.
        3. Block new entries.
        """
        try:
            # Direct API request (not via WS)
            positions = self.client.get_position_risk(symbol=self.symbol)
            if not positions: return

            position_data = positions[0]
            amt = float(position_data.get('positionAmt', '0.0'))
            entry_p = float(position_data.get('entryPrice', '0.0'))

            # 1. If position is closed on Exchange but active in Bot -> Reset
            if amt == 0 and self.current_position_on_exchange:
                self.logger.warning("♻️ SYNC: Position is closed on Exchange but active in Bot. Resetting state.")
                self.current_position_on_exchange = None
                self.current_position_quantity = 0.0
                self._cancel_all_orders_for_symbol() # Clean old stops
                return

            # 2. If position EXISTS but Bot DOES NOT KNOW (Ghost Position)
            if amt != 0 and self.current_position_on_exchange is None:
                # --- PATCH START ---
                # If traps are set, one likely triggered.
                # Do NOT sync yet, wait for ORDER_TRADE_UPDATE to execute strategy logic.
                if self.trap_order_up is not None or self.trap_order_down is not None:
                    return
                # --- PATCH END ---

                # If found but not detected via Iron Grip
                side = 'LONG' if amt > 0 else 'SHORT'
                self.logger.warning(f"⚠️ SYNC TRIGGERED: Found {side} {amt} position via ACCOUNT_UPDATE.")
                
                # Restore state
                self.current_position_on_exchange = side
                self.current_position_quantity = abs(amt)
                self.current_position_entry_price = Decimal(str(entry_p))
                
                # Clear entry traps because we are in the market!
                self.trap_order_up = None
                self.trap_order_down = None
                
                # Check if stops exist on exchange
                open_orders = self.client.get_open_orders(symbol=self.symbol)
                has_stop = any(o['type'] in ['STOP_MARKET', 'STOP_LOSS', 'STOP'] for o in open_orders)
                has_take = any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in open_orders)

                if not has_stop or not has_take:
                    self.logger.warning("🛡️ GHOST POSITION HAS NO STOPS! Placing EMERGENCY SL/TP.")
                    
                    # Emergency Params (0.5%)
                    price_dec = self.current_position_entry_price
                    gap = price_dec * Decimal('0.005')
                    
                    if side == 'LONG':
                        sl = price_dec - gap
                        tp = price_dec + gap
                    else:
                        sl = price_dec + gap
                        tp = price_dec - gap
                    
                    tick = self.symbol_trading_rules.get('tickSize', '0.01')
                    prec = self.symbol_trading_rules.get('pricePrecision', 2)
                    
                    self._place_sl_tp_orders_after_entry(side, abs(amt), price_dec, sl, tp, prec, tick)
                else:
                    self.logger.info("🛡️ Ghost position already has open orders. Assuming safe.")

        except Exception as e:
            self.logger.error(f"Sync Error: {e}")

    def _prepare_ai_features(self, df_main):
        """
        PREPARE FEATURES FOR AI (IDENTICAL TO TRAINING).
        """
        if len(df_main) < 100: return None
        
        df = df_main.copy()
        
        # 1. Trend Slope
        shifted_close = df['close'].shift(48).fillna(df['close'])
        df['trend_slope'] = ((df['close'] - shifted_close) / (shifted_close + 1e-9) * 100).fillna(0)
        
        # 2. Log Returns
        prev_close = df['close'].shift(1).fillna(df['close']).replace(0, 1e-9)
        df['log_ret'] = np.log(df['close'] / prev_close).fillna(0) * 100
        
        # 3. Relative Volume
        df['vol_ma'] = df['volume'].rolling(50).mean().fillna(1.0).replace(0, 1.0)
        df['vol_rel'] = (df['volume'] / df['vol_ma']).fillna(0)
        
        # 4. Position in Range
        df['rolling_low'] = df['low'].rolling(96).min().fillna(df['low'])
        df['rolling_high'] = df['high'].rolling(96).max().fillna(df['high'])
        range_span = (df['rolling_high'] - df['rolling_low']).replace(0, 1.0)
        df['pos_in_range'] = ((df['close'] - df['rolling_low']) / range_span).fillna(0.5)
        
        # Matrix
        feature_matrix = np.zeros((len(df), 26))
        feature_matrix[:, 0] = df['log_ret'].values
        feature_matrix[:, 1] = df['trend_slope'].values
        feature_matrix[:, 2] = df['vol_rel'].values
        feature_matrix[:, 3] = df['pos_in_range'].values
        
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Last window [96, 26]
        last_window = feature_matrix[-96:]
        if len(last_window) < 96: return None
        
        return torch.FloatTensor(last_window).unsqueeze(1)

    def _check_and_execute_trade_logic(self, df):
        """
        FINAL SNIPER LOGIC.
        Fixed: Correct variable naming for atr_series.
        """
        if df.empty or self.ai_model is None: return

        # 0. BLOCK RE-ENTRY ON SAME CANDLE
        current_candle_timestamp = df.index[-1]
        if self.last_trade_candle_time == current_candle_timestamp:
            return

        # 0. Check Positions
        if int(time.time()) % 10 == 0: self._manual_position_sync()
        if self.current_position_on_exchange: return 

        # --- 1. INDICATORS CALCULATION (CRITICAL FOR X) ---
        # Ми повинні порахувати X перед тим, як давати дані нейронці
        if self.klines_df_higher_tf.empty:
            return

        # Рахуємо X
        df_with_indicators = self._calculate_indicators(df.copy(), self.klines_df_higher_tf.copy())
        
        if df_with_indicators.empty:
             return

        # --- 2. PHYSICS CALCULATION ---
        # Тепер передаємо DF з індикаторами. 
        # Функція повертає (Фічі, ATR_для_Стопів).
        # 🔥 ВАЖЛИВО: Називаємо другу змінну atr_series, щоб код нижче (рядок 1995) не ламався!
        features_list, atr_series = self.physics_engine.process_window(df_with_indicators)
        
        # Discard None values (warmup period)
        valid_feats = [f for f in features_list if f is not None]
        
        # Model requires exactly 64 candles of history
        if len(valid_feats) < 64: return

        # 2. SCALING & TENSOR
        try:
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(valid_feats)
            data_scaled = np.nan_to_num(data_scaled)
            
            # Last window [64, 23]
            last_seq = data_scaled[-64:]
            input_tensor = torch.FloatTensor(last_seq).unsqueeze(0).to(self.device)
        except Exception as e:
            self.logger.error(f"Scaling Error: {e}")
            return

        # 3. AI PREDICTION
        try:
            with torch.no_grad():
                logits = self.ai_model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                
            long_prob = probs[0, 1].item()
            short_prob = probs[0, 2].item()
            
            # Control log (once per minute)
            self.logger.info(f"📊 AI SCAN: Long {long_prob:.2f} | Short {short_prob:.2f} | (Threshold: {self.CONFIDENCE_THRESHOLD})")

        except Exception as e:
            self.logger.error(f"AI Error: {e}")
            return

        # 4. DECISION (Threshold 0.70)
        signal = 0
        if long_prob > self.CONFIDENCE_THRESHOLD: signal = 1
        elif short_prob > self.CONFIDENCE_THRESHOLD: signal = 2
        
        if signal == 0: return

        # 5. EXECUTION (NO MERCY)
        current_price = float(df['close'].iloc[-1])
        current_atr = float(atr_series[-1])
        
        self.logger.info(f"🚀 SNIPER SIGNAL: {'LONG' if signal == 1 else 'SHORT'} (Conf: {max(long_prob, short_prob):.2f})")

        # --- 🛡️ SMART LEVERAGE (FIXED) ---
        # 1. Рахуємо дистанцію стопу
        sl_mult = getattr(self, 'SL_ATR_MULT', 1.0)
        sl_dist_value = sl_mult * current_atr
        sl_dist_pct = sl_dist_value / current_price if current_price > 0 else 0.01

        # 2. Рахуємо безпечне плече (щоб стоп з'їдав макс 90% маржі, а не ліквідував)
        max_safe_risk_on_equity = 0.90
        if sl_dist_pct > 0:
            safe_leverage = int(max_safe_risk_on_equity / sl_dist_pct)
        else:
            safe_leverage = 20

        # 3. Обмежуємо хард-капом (15x)
        # 3. Обмежуємо хард-капом (15x)
        target_leverage_cap = 15
        
        # 🔥 FIX: MIN NOTIONAL BOOST (Для малих депозитів < 5$)
        # Binance вимагає мінімум 5$ в угоді.
        balance_check = self._get_account_balance_usdt()
        if balance_check and balance_check > 0:
            risk_val = getattr(self, 'RISK_PER_TRADE', 0.95)
            # Скільки ми реально ставимо маржі (наприклад 1.3$)
            margin_used = balance_check * risk_val
            
            # Мінімальне плече, щоб пробити 5.1$ (наприклад: 5.1 / 1.3 = 3.92 -> 4x)
            if margin_used < 5.1:
                min_required_lev = 5.1 / margin_used
                required_lev_int = int(min_required_lev) + 1
                
                # Якщо "безпечне" плече менше за те, що вимагає біржа -> беремо біржове
                if safe_leverage < required_lev_int:
                    self.logger.warning(f"⚠️ LEVERAGE BOOST: Safe {safe_leverage}x -> Boosted {required_lev_int}x (to hit MinNotional 5$)")
                    safe_leverage = required_lev_int

        # Фінальне плече (не менше 1 і не більше 20, бо 15 може не вистачити для 1$)
        # Дозволяємо до 20x, якщо треба пробити мінімалку
        target_leverage = int(max(1, min(20, safe_leverage)))

        self.logger.info(f"🛡️ SAFETY: ATR={current_atr:.4f} ({sl_dist_pct*100:.2f}%). Safe Lev: {safe_leverage}x. Active Lev: {target_leverage}x")

        # --- POSITION SIZE CALCULATION ---
        balance = self._get_account_balance_usdt()
        if not balance or balance <= 0: 
            self.logger.error("❌ Balance is 0 or unreadable.")
            return

        # Встановлюємо плече
        try: self.client.change_leverage(symbol=self.symbol, leverage=target_leverage)
        except: pass

        # RISK: 95% of deposit * Leverage
        # Тепер target_leverage визначено, і помилки не буде
        risk_factor = getattr(self, 'RISK_PER_TRADE', 0.95) 
        position_value = balance * risk_factor * target_leverage
        quantity = position_value / current_price
        
        # Rounding to exchange step size
        step_size = self.symbol_trading_rules.get('stepSize', '1')
        quantity = self._round_quantity_to_step_size(quantity, step_size)
        
        # Check MinNotional
        min_notional = float(self.symbol_trading_rules.get('minNotional', 5.0))
        if (quantity * current_price) < min_notional:
            self.logger.warning(f"❌ Order value < MinNotional")
            return

        side = 'BUY' if signal == 1 else 'SELL'
        pos_side = 'LONG' if signal == 1 else 'SHORT'
        
        # --- STOP LOSS / TAKE PROFIT CALCULATION (2 / 6 ATR) ---
        sl_price = 0
        tp_price = 0
        
        # Use parameters from __init__
        sl_mult = getattr(self, 'SL_ATR_MULT', 1.0)
        tp_mult = getattr(self, 'TP_ATR_MULT', 3.0)

        if signal == 1: # Long
            sl_price = current_price - (sl_mult * current_atr)
            tp_price = current_price + (tp_mult * current_atr)
        else: # Short
            sl_price = current_price + (sl_mult * current_atr)
            tp_price = current_price - (tp_mult * current_atr)

        # 6. SENDING ORDER (LIMIT ENTRY)
        try:
            # Format price for Limit Order
            tick_size = self.symbol_trading_rules.get('tickSize', '0.01')
            prec = self.symbol_trading_rules.get('pricePrecision', 2)
            
            # Enter at candle close price (Signal Price)
            # To guarantee execution, we place a LIMIT at current price
            limit_price_dec = self._round_price_to_tick_size(Decimal(str(current_price)), tick_size)
            formatted_price = f"{limit_price_dec:.{prec}f}"

            self.logger.info(f"⚡ PLACING LIMIT {side} {quantity} @ {formatted_price} (Lev x{target_leverage})...")
            
            order = self.client.new_order(
                symbol=self.symbol,
                side=side,
                positionSide=pos_side,
                type='LIMIT',       # LIMIT ENTRY
                timeInForce='GTC',  # Good Till Cancel
                quantity=quantity,
                price=formatted_price
            )
            
            # For Limit orders, we use the limit price as the base for stops
            entry_p = float(formatted_price)
            
            # Place Stops (Separate Orders)
            # The method _place_sl_tp... handles LIMIT TP and STOP MARKET SL
            # self._place_sl_tp_orders_after_entry(
            #     position_side=pos_side,
            #     position_qty=quantity,
            #     entry_price=Decimal(str(entry_p)),
            #     sl_price_from_signal=Decimal(str(sl_price)),
            #     tp_price_from_signal=Decimal(str(tp_price)),
            #     price_precision=prec,
            #     tick_size=tick_size
            # )
            
            # Cooldown to prevent re-entry on the same candle
            # self.pending_ai_trade_details = {'timestamp': time.time()} # <-- БУЛО ТАК (ПОМИЛКА)
            
            # 🔥 FIX: ЗБЕРІГАЄМО РОЗРАХОВАНІ % СТОПІВ, ЩОБ IRON GRIP ЇХ ПОБАЧИВ
            sl_dist_pct = (sl_mult * current_atr) / current_price
            tp_dist_pct = (tp_mult * current_atr) / current_price
            
            self.pending_ai_trade_details = {
                'timestamp': time.time(),
                'sl_pct': float(sl_dist_pct),  # Передаємо порахований ATR стоп
                'tp_pct': float(tp_dist_pct)
            }
            self.logger.info(f"💾 CONTEXT SAVED: SL {sl_dist_pct*100:.2f}%, TP {tp_dist_pct*100:.2f}%")
            
        except Exception as e:
            self.logger.error(f"❌ EXECUTION FAILED: {e}")

    def _update_or_create_trap_order(self, trap_side, side, price, x_val):
        """
        CREATE TRAP (Real Orders).
        Fixed: Error handling for -2021 (Immediate Trigger) -> Conversion to MARKET Order.
        """
        # 1. Cancel old trap
        existing_order = self.trap_order_up if trap_side == 'UP' else self.trap_order_down
        
        if existing_order:
            old_price = float(existing_order['price'])
            new_price = float(price)
            if abs(old_price - new_price) < (float(x_val) * 0.05): return 
            
            self.logger.info(f"Trap {trap_side} moved. Cancel old {old_price} -> New {new_price}")
            self._cancel_order(existing_order['orderId'])
            
            if trap_side == 'UP': self.trap_order_up = None
            else: self.trap_order_down = None

        # 2. Calculations
        entry_side = 'LONG' if side == 'BUY' else 'SHORT'
        price_dec = Decimal(str(price))
        x_dec = Decimal(str(x_val))
        
        tp_dist = x_dec * self.tp_coeff
        sl_dist = x_dec * self.sl_coeff
        
        if entry_side == 'LONG':
            sl_price = price_dec - sl_dist
            tp_price = price_dec + tp_dist
        else:
            sl_price = price_dec + sl_dist
            tp_price = price_dec - tp_dist

        # 3. Determine Type (Limit vs Breakout)
        try:
            current_market_price = float(self.client.ticker_price(symbol=self.symbol)['price'])
        except:
            return

        is_breakout = False
        if side == 'BUY' and float(price) > current_market_price: is_breakout = True
        if side == 'SELL' and float(price) < current_market_price: is_breakout = True

        # 4. Risk Management
        price_move_to_sl_abs = abs(price_dec - sl_price)
        if price_dec <= 0 or price_move_to_sl_abs <= 0: return
        
        target_risk_pct = Decimal(str(self.target_percent_param)) / Decimal('100')
        price_move_to_sl_pct = (price_move_to_sl_abs / price_dec)
        required_leverage = max(1, int(target_risk_pct / price_move_to_sl_pct)) if price_move_to_sl_pct > 0 else 1
        
        notional_check = float(self.virtual_bot_capital_usdt * Decimal(str(required_leverage)))
        max_lev = self._get_max_allowed_leverage_for_notional(notional_check)
        final_leverage = min(required_leverage, max_lev)
        try: self.client.change_leverage(symbol=self.symbol, leverage=final_leverage)
        except: pass
        
        position_notional = (self.virtual_bot_capital_usdt * Decimal(str(final_leverage))) * Decimal('0.49')
        quantity_dec = position_notional / price_dec
        quantity = self._round_quantity_to_step_size(float(quantity_dec), self.symbol_trading_rules.get('stepSize'))
        
        final_notional = Decimal(str(quantity)) * price_dec
        if final_notional < Decimal(str(self.symbol_trading_rules.get('minNotional', '5.0'))):
            return

        self.logger.info(f"💵 REINVEST: Cap {self.virtual_bot_capital_usdt:.2f} USDT -> Size {quantity} {self.symbol} (Lev {final_leverage}x)")
        
        # 5. SEND ORDER
        cl_id = f"trap-{trap_side}-{int(time.time()*1000)}"
        order = None
        order_type_log = ""
        
        safe_price_base = self._round_price_to_tick_size(price_dec, self.symbol_trading_rules.get('tickSize', '0.01'))
        price_str = f"{safe_price_base:.{self.symbol_trading_rules.get('pricePrecision', 2)}f}"

        try:
            if not is_breakout:
                # --- LIMIT (Pullback) ---
                order_type_log = "LIMIT"
                r_side = 'FLOOR' if side == 'BUY' else 'CEIL'
                safe_price = self._round_price_to_tick_size(price_dec, self.symbol_trading_rules.get('tickSize', '0.01'), side=r_side)
                order = self._place_limit_order(side, quantity, float(safe_price), entry_side, cl_id)
            else:
                # --- ALGO STOP_MARKET (Breakout) ---
                order_type_log = "ALGO STOP"
                self.logger.info(f"🚀 Placing ALGO BREAKOUT Trap: {side} {quantity} @ {price_str}")
                
                params = {
                    'symbol': self.symbol,
                    'side': side,
                    'positionSide': entry_side,
                    'algoType': 'CONDITIONAL',
                    'type': 'STOP_MARKET',
                    'quantity': quantity,
                    'triggerPrice': price_str,
                    'workingType': 'MARK_PRICE',
                    'priceProtect': 'TRUE',
                    'newClientOrderId': cl_id
                }
                
                # Catching -2021 error here
                order = self._send_algo_request('POST', '/fapi/v1/algoOrder', params)
                
                if order and ('algoId' in order or 'clientAlgoId' in order):
                    order['orderId'] = order.get('algoId')

        except ClientError as e:
            # !!! CRITICAL FIX: If price already passed the level (-2021) -> Enter MARKET !!!
            if e.error_code == -2021:
                self.logger.warning(f"⚡ IMMEDIATE TRIGGER (-2021) detected! Breakout confirmed. Executing MARKET {side} immediately!")
                try:
                    # Formulate standard Market Order
                    market_params = {
                        'symbol': self.symbol,
                        'side': side,
                        'positionSide': entry_side,
                        'type': 'MARKET',
                        'quantity': quantity
                    }
                    order = self.client.new_order(**market_params)
                    order_type_log = "MARKET (Recovered)"
                    self.logger.info(f"✅ MARKET Recovery successful. OrderID: {order.get('orderId')}")
                except Exception as market_e:
                    self.logger.error(f"❌ Failed to execute Market Recovery: {market_e}")
                    return
            else:
                self.logger.error(f"❌ Trap ClientError ({order_type_log}): {e}")
                return

        except Exception as e:
            self.logger.error(f"❌ Trap Error ({order_type_log}): {e}")
            return

        # 6. Save
        if order:
            # For Market order orderId is standard, for Algo - algoId (handled above)
            order_info = {
                "orderId": str(order['orderId']),
                "clientOrderId": cl_id,
                "price": price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "position_side": entry_side,
                "type": order_type_log,
                "is_algo": (order_type_log == "ALGO STOP"),
                "price_precision_asset": self.symbol_trading_rules.get('pricePrecision'),
                "tick_size_str": self.symbol_trading_rules.get('tickSize')
            }
            
            if trap_side == 'UP': self.trap_order_up = order_info
            else: self.trap_order_down = order_info
            
            self.logger.info(f"action: new entry TRAP {trap_side} SET ({order_type_log}): {side} {quantity} @ {price}")

    def _cancel_single_trap(self, trap_side):
        target = self.trap_order_up if trap_side == 'UP' else self.trap_order_down
        if target:
            self.logger.info(f"Removing {trap_side} trap order {target['orderId']}")
            
            self._cancel_order(target['orderId'])
            
            if trap_side == 'UP': self.trap_order_up = None
            else: self.trap_order_down = None

    def _cancel_trap_orders(self):
        if self.trap_order_up:
            self._cancel_order(self.trap_order_up['orderId'])
            self.trap_order_up = None
        if self.trap_order_down:
            self._cancel_order(self.trap_order_down['orderId'])
            self.trap_order_down = None
    

    def _execute_limit_entry(self, entry_side, entry_price, sl_price, tp_price, current_step_val, x_val):
        """
        Helper method to send entry order.
        Separated to keep _check_and_execute_trade_logic cleaner.
        """
        try:
            # 1. Risk and Leverage Calculation
            price_move_to_sl_abs = abs(entry_price - sl_price)
            if entry_price <= 0 or price_move_to_sl_abs <= 0: return
            
            target_risk_pct = Decimal(str(self.target_percent_param)) / Decimal('100')
            price_move_to_sl_pct = (price_move_to_sl_abs / entry_price)
            
            required_leverage = int(target_risk_pct / price_move_to_sl_pct)
            if required_leverage < 1: required_leverage = 1
            
            # Check max notional
            notional_check = float(self.virtual_bot_capital_usdt * Decimal(str(required_leverage)))
            max_lev = self._get_max_allowed_leverage_for_notional(notional_check)
            final_leverage = min(required_leverage, max_lev)
            
            self.client.change_leverage(symbol=self.symbol, leverage=final_leverage)
            
            # 2. Quantity Calculation
            position_notional = (self.virtual_bot_capital_usdt * Decimal(str(final_leverage))) * Decimal('0.98')
            quantity_dec = position_notional / entry_price
            quantity = self._round_quantity_to_step_size(float(quantity_dec), self.symbol_trading_rules.get('stepSize'))
            
            min_notional = Decimal(str(self.symbol_trading_rules.get('minNotional', '5.0')))
            if (Decimal(str(quantity)) * entry_price) < min_notional:
                self.logger.warning(f"Quantity {quantity} too small for minNotional. Skip.")
                return

            # 3. Order Sending
            tick_size = self.symbol_trading_rules.get('tickSize')
            side = 'BUY' if entry_side == 'LONG' else 'SELL'
            
            # Note: For HL logic we want to enter as soon as the level changes.
            # Using LIMIT at entry_price.
            
            new_cl_id = f"hl-{int(time.time()*1000)}"
            
            order = self._place_limit_order(side, quantity, float(entry_price), entry_side, new_cl_id)
            
            if order:
                self.active_limit_entry_order_details.update({
                    "orderId": str(order['orderId']),
                    "clientOrderId": new_cl_id,
                    "symbol": self.symbol,
                    "order_side": side,
                    "position_side": entry_side,
                    "price": float(entry_price),
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "status": "NEW",
                    "step_level_at_signal": current_step_val,
                    "price_precision_asset": self.symbol_trading_rules.get('pricePrecision'),
                    "tick_size_str": tick_size
                })
                
        except Exception as e:
            self.logger.error(f"Execution Error: {e}", exc_info=True)

    def _send_algo_request(self, method, path, params=None):
        """
        Universal method for Algo orders.
        Fixed: Propagates error -2021 (Immediate Trigger) to handle it as Market entry.
        """
        if params is None: params = {}
        try:
            response = self.client.sign_request(method, path, params)
            return response
        except ClientError as e:
            # -2021: Order would immediately trigger. 
            # This means price already crossed the level. We need to handle this above.
            if e.error_code == -2021:
                raise e 
            
            # Log and ignore other errors
            error_msg = str(e)
            if "-2011" in error_msg or "-2013" in error_msg: # Unknown order / Order does not exist
                return None
                
            self.logger.error(f"Algo API ClientError [{method} {path}]: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Algo API Request Failed [{method} {path}]: {e}")
            return None

    def _place_limit_order(self, side: str, quantity: float, price: float, position_side: str, new_client_order_id: str):
        if quantity <= 0 or price <= 0:
            self.logger.warning(f"Attempted to place LIMIT order for {self.symbol} with invalid quantity {quantity} or price {price}. Order not placed.")
            return None

        price_precision = self.symbol_trading_rules.get('pricePrecision', 2) 
        formatted_price = f"{price:.{price_precision}f}"

        self.logger.info(f"Attempting to place LIMIT order for {self.symbol}: {side} {quantity} at price {formatted_price}, positionSide: {position_side}")
        try:
            order_params = {
                'symbol': self.symbol,
                'side': side,
                'positionSide': position_side,
                'type': 'LIMIT',
                'quantity': quantity, 
                'price': formatted_price,
                'timeInForce': 'GTC',
                'newClientOrderId': new_client_order_id 
            }
            order = self.client.new_order(**order_params)
            self.logger.info(f"LIMIT order request for {self.symbol} sent successfully. Response: {order}")
            return order
        except ClientError as e:
            self.logger.error(f"API ClientError placing LIMIT order for {self.symbol} {side} {quantity} @ {formatted_price}: "
                              f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.header}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error placing LIMIT order for {self.symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error placing LIMIT order for {self.symbol}: {e}", exc_info=True)
        return None

    def _cancel_order(self, order_id):
        """
        Universal Order Cancel (Smart Cancel).
        Replaces the old method.
        First tries to cancel as Algo Order (since we use them now),
        if fails — tries as Standard Order.
        """
        if not order_id:
            return

        # Attempt 1: Cancel as Algo Order (Conditional)
        try:
            params = {'symbol': self.symbol, 'algoId': order_id}
            resp = self._send_algo_request('DELETE', '/fapi/v1/algoOrder', params)
            
            if resp and ('algoId' in resp or resp.get('code') == 200):
                self.logger.info(f"✅ Algo order {order_id} canceled successfully.")
                return
        except Exception:
            # If error (e.g. not an Algo order), proceed to Plan B
            pass 

        # Attempt 2: Cancel as Standard Order (Limit/Market)
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
            self.logger.info(f"✅ Standard order {order_id} canceled successfully.")
        except Exception as e:
            error_msg = str(e)
            if "-2011" not in error_msg and "Unknown order" not in error_msg:
                self.logger.warning(f"Failed to cancel order {order_id}: {e}")
    
    def _cancel_all_orders_for_symbol(self):
        """
        Cancels ALL orders: Standard and ALGO (Conditional).
        Added RETRY mechanism for reliability during network glitches.
        """
        max_retries = 3
        
        # 1. Cancel Standard Orders (Limit/Market)
        for i in range(max_retries):
            try:
                self.client.cancel_open_orders(symbol=self.symbol)
                self.logger.info(f"All STANDARD open orders for {self.symbol} canceled.")
                break 
            except Exception as e:
                self.logger.warning(f"Failed to cancel standard orders (Attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    time.sleep(0.5) 
                else:
                    self.logger.error(f"CRITICAL: Failed to cancel STANDARD orders after {max_retries} attempts!")

        # 2. Cancel ALGO Orders (STOP/TP)
        for i in range(max_retries):
            try:
                # Use special endpoint for mass Algo cancel
                self._send_algo_request('DELETE', '/fapi/v1/algoOpenOrders', {'symbol': self.symbol})
                self.logger.info(f"All ALGO open orders for {self.symbol} canceled.")
                break
            except Exception as e:
                self.logger.warning(f"Failed to cancel ALGO orders (Attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    time.sleep(0.5)
                else:
                    self.logger.error(f"CRITICAL: Failed to cancel ALGO orders after {max_retries} attempts!")
            
        # Clear local memory
        self.trap_order_up = None
        self.trap_order_down = None
    
    def _round_price_to_tick_size(self, price, tick_size_str: str, side: str = "DEFAULT"):
        # Fixed: Function now always returns Decimal to preserve precision
        if not tick_size_str:
            self.logger.warning(f"Invalid tick_size_str (empty) for price rounding on {self.symbol}. Returning raw price as Decimal.")
            return Decimal(str(price))
        try:
            tick_size_float = float(tick_size_str)
            if tick_size_float <= 0:
                self.logger.warning(f"Invalid tick_size_str (<=0) for price rounding on {self.symbol}: '{tick_size_str}'. Returning raw price as Decimal.")
                return Decimal(str(price))
        except ValueError:
            self.logger.warning(f"Invalid tick_size_str (not a float) for price rounding on {self.symbol}: '{tick_size_str}'. Returning raw price as Decimal.")
            return Decimal(str(price))

        price_dec = Decimal(str(price))
        tick_dec = Decimal(tick_size_str)

        if '.' in tick_size_str:
            num_decimals = len(tick_size_str.split('.')[1])
        else:
            num_decimals = 0

        rounding_mode = ROUND_DOWN
        if side == "SL_LONG" or side == "TP_SHORT":
            rounding_mode = ROUND_DOWN
        elif side == "SL_SHORT" or side == "TP_LONG":
            rounding_mode = ROUND_UP
        else: # DEFAULT rounding
             # self.logger.warning(f"Rounding price with DEFAULT side for {self.symbol}. This may lead to suboptimal execution. Please specify rounding side.")
             rounding_mode = ROUND_DOWN # Safe default
        
        rounded_price_dec = (price_dec / tick_dec).to_integral_value(rounding=rounding_mode) * tick_dec
        
        return rounded_price_dec.quantize(Decimal('1e-' + str(num_decimals)))
    

    def _ensure_all_orders_canceled_for_shutdown(self):
        self.logger.info(f"FORCE CANCEL ALL ORDERS for {self.symbol} before shutdown...")
        
        # 1. Cancel via API
        try:
            self.client.cancel_open_orders(symbol=self.symbol)
            self.logger.info(f"All orders for {self.symbol} successfully canceled via API.")
        except ClientError as e:
            if e.error_code == -2011:
                self.logger.info(f"No orders on exchange for {self.symbol}.")
            else:
                self.logger.error(f"API Error canceling all orders: {e}")
        except Exception as e:
            self.logger.error(f"Unknown error during mass cancel: {e}")

        # 2. Clear local variables
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.trap_order_up = None
        self.trap_order_down = None

        self._reset_active_limit_entry_order_details()
        
        time.sleep(1.0) 
        self.logger.info(f"Active order check for {self.symbol} completed.")

    def _close_open_position_on_shutdown(self):
        """Checks for open position and closes it with LIMIT order on shutdown."""
        self.logger.info(f"Checking for open position for {self.symbol} before shutdown...")
        
        try:
            # Fresh request to exchange
            positions = self._get_open_positions(symbol_to_check=self.symbol, retries=1)
            
            if positions: 
                position_info = positions[0]
                position_amt = float(position_info.get('positionAmt', 0))
                
                if position_amt != 0:
                    self.logger.warning(
                        f"Open position detected: {position_amt} {self.symbol}. "
                        f"Initiating forced LIMIT close."
                    )
                    
                    # First cancel all orders
                    self._ensure_all_orders_canceled_for_shutdown()
                    time.sleep(1) 

                    # Determine closing side
                    closing_side = "SELL" if position_amt > 0 else "BUY"
                    position_side_to_close = "LONG" if position_amt > 0 else "SHORT"
                    
                    # Place Limit close order
                    close_order = self._place_aggressive_limit_order_to_close(
                        order_closing_side=closing_side,
                        quantity=abs(position_amt),
                        position_being_closed_side=position_side_to_close
                    )
                    
                    if close_order and close_order.get('orderId'):
                        self.logger.info(f"Limit close order for {self.symbol} sent successfully.")
                        time.sleep(3) # Wait for execution
                    else:
                        self.logger.error(f"FAILED to close position for {self.symbol} on shutdown! MANUAL INTERVENTION REQUIRED!")
                else:
                    self.logger.info(f"No active positions for {self.symbol} detected on exchange.")
            else:
                self.logger.info(f"Could not get position info or none exists.")

        except Exception as e:
            self.logger.error(f"Error during forced position close for {self.symbol}: {e}", exc_info=True)

    def _calculate_quantity(self, price, capital, leverage,
                           step_size_str: str, min_qty_val: float, qty_precision_from_rules: int):
        # FIXED: Convert inputs to Decimal for safe calculations
        try:
            price_dec = Decimal(str(price))
            capital_dec = Decimal(str(capital))
            leverage_dec = Decimal(str(leverage))
        except Exception as e:
            self.logger.error(f"Error converting inputs to Decimal in _calculate_quantity for {self.symbol}: {e}")
            return 0.0

        if price_dec <= 0:
            self.logger.error(f"Invalid input for _calculate_quantity for {self.symbol}: price={price_dec}")
            return 0.0

        if not step_size_str:
             self.logger.error(f"Invalid step_size_str (empty) for _calculate_quantity for {self.symbol}")
             return 0.0

        try:
            position_value_usdt = capital_dec * leverage_dec
            raw_quantity = position_value_usdt / price_dec

            # _round_quantity_to_step_size already returns float, this is correct
            final_quantity = self._round_quantity_to_step_size(raw_quantity, step_size_str)

            self.logger.debug(f"CalcQty ({self.symbol}): raw_qty={raw_quantity:.8f}, capital={capital_dec}, lev={leverage_dec}, price={price_dec} -> final_qty={final_quantity:.{qty_precision_from_rules}f}")

            if final_quantity < min_qty_val:
                self.logger.warning(f"CalcQty ({self.symbol}): Rounded quantity {final_quantity:.{qty_precision_from_rules}f} < minQty {min_qty_val}. Returning 0.")
                return 0.0
            return final_quantity
        except Exception as e:
            self.logger.error(f"Error calculating quantity for {self.symbol}: {e}", exc_info=True)
            return 0.0
            
    def _round_quantity_to_step_size(self, quantity: float, step_size_str: str):
        if not step_size_str:
             self.logger.warning(f"Invalid step_size_str (empty) for quantity rounding on {self.symbol}. Returning raw quantity.")
             return quantity
        try:
            step_size_float = float(step_size_str)
            if step_size_float <= 0:
                self.logger.warning(f"Invalid step_size_str (<=0) for quantity rounding on {self.symbol}: '{step_size_str}'. Returning raw quantity.")
                return quantity
        except ValueError:
            self.logger.warning(f"Invalid step_size_str (not a float) for quantity rounding on {self.symbol}: '{step_size_str}'. Returning raw quantity.")
            return quantity

        quantity_dec = Decimal(str(quantity))
        step_dec = Decimal(step_size_str)
        # Round down to nearest step size
        rounded_qty_dec = (quantity_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
        
        # Determine decimal places from step_size_str
        if '.' in step_size_str:
            num_decimals = len(step_size_str.split('.')[1])
        else:
            num_decimals = 0
            
        return float(rounded_qty_dec.quantize(Decimal('1e-' + str(num_decimals))))


    def _get_max_allowed_leverage_for_notional(self, notional_value: float):
        if not self.leverage_brackets_for_symbol:
            self.logger.warning(f"Leverage bracket data for {self.symbol} is not available. Returning default max leverage (e.g., 20x).")
            return 20

        sorted_brackets = sorted(self.leverage_brackets_for_symbol, key=lambda x: float(x.get('notionalFloor', 0)))
        
        # Iterate from largest brackets to smallest
        for bracket in reversed(sorted_brackets):
            floor = float(bracket.get('notionalFloor', 0))
            if notional_value >= floor:
                return int(bracket.get('initialLeverage', 1))

        if sorted_brackets:
             self.logger.warning(f"Could not determine specific leverage bracket for notional {notional_value} for {self.symbol}. Using leverage from the smallest notional bracket.")
             return int(sorted_brackets[0].get('initialLeverage', 1))
        
        self.logger.warning(f"Could not determine appropriate leverage bracket for notional value {notional_value} USDT for {self.symbol}. Returning 1x for safety.")
        return 1

 
    def _place_sl_tp_orders_after_entry(self, position_side: str, position_qty: float,
                                        entry_price: Decimal,
                                        sl_price_from_signal: Decimal, 
                                        tp_price_from_signal: Decimal, 
                                        price_precision: int, tick_size: str,
                                        liquidation_price: Decimal = Decimal('0')):
        
        if entry_price <= 0 or sl_price_from_signal <= 0:
            self.logger.warning(f"⚠️ SL/TP Placement skipped: Invalid prices.")
            return

        # 1. CLEANUP
        try:
            self.logger.info(f"🧹 CLEANUP: Removing ALL existing orders before placing new ones.")
            self._send_algo_request('DELETE', '/fapi/v1/algoOpenOrders', {'symbol': self.symbol})
            self.client.cancel_open_orders(symbol=self.symbol)
            time.sleep(0.5) # Wait for cancel to propagate
        except Exception as e:
            self.logger.warning(f"Cleanup Warning: {e}")

        # 2. CALC PRICES
        self.logger.info(f"Placing SL/TP for {position_side} {position_qty} {self.symbol}")
        
        sl_price_raw = sl_price_from_signal
        tp_price_raw = tp_price_from_signal

        sl_order_side, tp_order_side = "", ""
        rounding_sl_side, rounding_tp_side = "", ""

        if position_side == 'LONG':
            sl_order_side, tp_order_side = "SELL", "SELL"
            rounding_sl_side, rounding_tp_side = "SL_LONG", "TP_LONG"
        elif position_side == 'SHORT':
            sl_order_side, tp_order_side = "BUY", "BUY"
            rounding_sl_side, rounding_tp_side = "SL_SHORT", "TP_SHORT"
        else:
            self.logger.error(f"Invalid position_side '{position_side}'")
            return

        sl_price_final = self._round_price_to_tick_size(sl_price_raw, tick_size, side=rounding_sl_side)
        tp_price_final = self._round_price_to_tick_size(tp_price_raw, tick_size, side=rounding_tp_side)
        
        self.logger.info(f"Calculated SL: {sl_price_final:.{price_precision}f}, TP: {tp_price_final:.{price_precision}f}")

        # ======================================================================
        # 3. PLACE TAKE PROFIT (LIMIT MAKER) - FIRST!
        # ======================================================================
        # We place TP first as a Limit order. It locks the quantity.
        tp_is_valid = True
        if tp_price_final <= 0: tp_is_valid = False

        if tp_is_valid and position_qty > 0:
            try:
                formatted_tp_price = f"{tp_price_final:.{price_precision}f}"
                self.logger.info(f"💰 Placing MAKER TP (Limit) @ {formatted_tp_price}")
                
                params_tp = {
                    'symbol': self.symbol,
                    'side': tp_order_side,       
                    'positionSide': position_side, 
                    'type': 'LIMIT',              
                    'quantity': float(position_qty), # Float required for standard new_order
                    'price': formatted_tp_price,
                    'timeInForce': 'GTC'
                }
                
                res_tp = self.client.new_order(**params_tp)
                
                if res_tp and res_tp.get('orderId'):
                    self.active_tp_order_id = str(res_tp.get('orderId'))
                    self.logger.info(f"✅ TP (Maker Limit) PLACED. ID: {self.active_tp_order_id}")
                else:
                    self.logger.warning(f"⚠️ TP Placement Failed.")
            except Exception as e:
                 self.logger.error(f"❌ TP LIMIT ERROR: {e}")

        # ======================================================================
        # 4. PLACE STOP LOSS (MARKET CLOSE ALL) - SECOND!
        # ======================================================================
        # We use closePosition='TRUE' to avoid "ReduceOnly" conflict with the TP order.
        sl_is_valid = True
        if sl_price_final <= 0: sl_is_valid = False

        if sl_is_valid:
            params_sl = {
                'symbol': self.symbol, 
                'side': sl_order_side, 
                'positionSide': position_side,
                'algoType': 'CONDITIONAL', 
                'type': 'STOP_MARKET', 
                # 'quantity': str(position_qty), <--- REMOVED QUANTITY
                'closePosition': 'TRUE',       # <--- ADDED CLOSE POSITION (Fixes conflict)
                'triggerPrice': f"{sl_price_final:.{price_precision}f}", 
                'workingType': 'MARK_PRICE', 
                'priceProtect': 'TRUE'
            }
            
            res_sl = self._send_algo_request('POST', '/fapi/v1/algoOrder', params_sl)
            
            if res_sl and (res_sl.get('clientAlgoId') or res_sl.get('orderId')):
                self.active_sl_order_id = str(res_sl.get('orderId', res_sl.get('clientAlgoId')))
                self.logger.info(f"✅ SL (Market Close) PLACED. ID: {self.active_sl_order_id}")
            else:
                self.logger.error(f"❌ SL API FAILED. Position UNPROTECTED!")
                # If SL fails, we should panic, but TP is already there at least.
                self._close_position_critically(reason="SL Placement Failed")

    def _place_stop_market_order(self, side: str, quantity: float, stop_price: Decimal, price_precision: int, position_side_to_close: str, rounding_side: str):
        """
        REAL ALGO STOP LOSS (STOP_MARKET).
        """
        if stop_price <= 0: return None

        try:
            stop_price_rounded = self._round_price_to_tick_size(stop_price, self.symbol_trading_rules.get('tickSize', '0.01'), side=rounding_side)
            trigger_price_str = f"{stop_price_rounded:.{price_precision}f}"

            self.logger.info(f"🛡️ ALGO SL: Placing STOP_MARKET for {self.symbol}. Trigger: {trigger_price_str}")

            params = {
                'symbol': self.symbol,
                'side': side,
                'positionSide': position_side_to_close,
                'algoType': 'CONDITIONAL',     
                'type': 'STOP_MARKET',         
                'quantity': str(quantity),
                'triggerPrice': trigger_price_str, 
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE'
            }

            order = self._send_algo_request('POST', '/fapi/v1/algoOrder', params)
            
            if order and ('algoId' in order or 'clientAlgoId' in order):
                algo_id = order.get('algoId')
                self.logger.info(f"✅ ALGO SL placed. AlgoID: {algo_id}")
                order['orderId'] = algo_id 
                return order
            else:
                self.logger.error("❌ ALGO SL Failed: No ID in response.")
                return None

        except Exception as e:
            self.logger.error(f"❌ ALGO SL EXCEPTION: {e}")
            return None

    def _place_take_profit_market_order(self, side: str, quantity: float, stop_price: Decimal, price_precision: int, position_side_to_close: str, rounding_side: str):
        """
        REAL ALGO TAKE PROFIT (TAKE_PROFIT_MARKET).
        """
        if stop_price <= 0: return None

        try:
            stop_price_rounded = self._round_price_to_tick_size(stop_price, self.symbol_trading_rules.get('tickSize', '0.01'), side=rounding_side)
            trigger_price_str = f"{stop_price_rounded:.{price_precision}f}"

            self.logger.info(f"💰 ALGO TP: Placing TP_MARKET for {self.symbol}. Trigger: {trigger_price_str}")

            params = {
                'symbol': self.symbol,
                'side': side,
                'positionSide': position_side_to_close,
                'algoType': 'CONDITIONAL',
                'type': 'TAKE_PROFIT_MARKET',
                'quantity': str(quantity),
                'triggerPrice': trigger_price_str,
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE'
            }

            order = self._send_algo_request('POST', '/fapi/v1/algoOrder', params)
            
            if order and ('algoId' in order or 'clientAlgoId' in order):
                algo_id = order.get('algoId')
                self.logger.info(f"✅ ALGO TP placed. AlgoID: {algo_id}")
                order['orderId'] = algo_id
                return order
            else:
                self.logger.error("❌ ALGO TP Failed: No ID in response.")
                return None

        except Exception as e:
            self.logger.error(f"❌ ALGO TP EXCEPTION: {e}")
            return None

              
    def _place_aggressive_limit_order_to_close(self, order_closing_side: str, quantity: float, position_being_closed_side: str):
        """
        REPLACES MARKET CLOSE.
        Places an aggressive LIMIT order to close position immediately.
        Aggressiveness: +/- 2 ticks from current ticker price.
        """
        if quantity <= 0:
            self.logger.warning(f"Attempted to place LIMIT CLOSE for {self.symbol} with quantity {quantity}. Not placed.")
            return None
            
        try:
            # 1. Get Ticker for Price
            ticker = self.client.ticker_price(symbol=self.symbol)
            market_price = float(ticker['price'])
            
            tick_size = float(self.symbol_trading_rules.get('tickSize', '0.01'))
            price_precision = int(self.symbol_trading_rules.get('pricePrecision', 2))
            
            # 2. Calculate Aggressive Limit Price
            # If we are Selling (closing Long), we want to sell LOWER (hit bid)
            # If we are Buying (closing Short), we want to buy HIGHER (hit ask)
            aggressive_ticks = 2
            
            limit_price = 0.0
            if order_closing_side == 'SELL':
                limit_price = market_price - (tick_size * aggressive_ticks)
            else:
                limit_price = market_price + (tick_size * aggressive_ticks)
                
            formatted_price = f"{limit_price:.{price_precision}f}"
            
            self.logger.info(f"Attempting AGGRESSIVE LIMIT CLOSE for {self.symbol}: {order_closing_side} {quantity} @ {formatted_price} (Mkt: {market_price})")
            
            # 3. Send Limit Order
            order_params = {
                'symbol': self.symbol, 
                'side': order_closing_side, 
                'positionSide': position_being_closed_side, 
                'type': 'LIMIT', 
                'timeInForce': 'GTC', # Good Till Cancel
                'quantity': quantity, 
                'price': formatted_price
            }
            
            order = self.client.new_order(**order_params)
            self.logger.info(f"LIMIT CLOSE order request for {self.symbol} sent. Response: {order}")
            return order

        except ClientError as e:
            self.logger.error(f"API ClientError placing LIMIT CLOSE for {self.symbol}: {e.error_message} (Code: {e.error_code})")
        except Exception as e:
            self.logger.error(f"Unexpected error placing LIMIT CLOSE for {self.symbol}: {e}", exc_info=True)
        return None

    def _place_market_order_to_close(self, order_closing_side: str, quantity: float, position_being_closed_side: str):
        """
        REPLACES MARKET CLOSE WITH AGGRESSIVE LIMIT CLOSE.
        Name kept as '_place_market_order_to_close' for compatibility with existing calls,
        but executes a LIMIT order +/- 2 ticks from price.
        """
        if quantity <= 0:
            self.logger.warning(f"Attempted to place CLOSE for {self.symbol} with quantity {quantity}. Not placed.")
            return None
            
        try:
            # 1. Get Ticker
            ticker = self.client.ticker_price(symbol=self.symbol)
            market_price = float(ticker['price'])
            
            tick_size = float(self.symbol_trading_rules.get('tickSize', '0.01'))
            price_precision = int(self.symbol_trading_rules.get('pricePrecision', 2))
            
            # 2. Calculate Aggressive Limit Price (+/- 2 ticks to ensure immediate fill)
            # If Sell -> Price - 2 ticks (Hit Bid)
            # If Buy -> Price + 2 ticks (Hit Ask)
            aggressive_ticks = 2
            
            limit_price = 0.0
            if order_closing_side == 'SELL':
                limit_price = market_price - (tick_size * aggressive_ticks)
            else:
                limit_price = market_price + (tick_size * aggressive_ticks)
                
            formatted_price = f"{limit_price:.{price_precision}f}"
            
            self.logger.info(f"Attempting LIMIT CLOSE (Aggressive) for {self.symbol}: {order_closing_side} {quantity} @ {formatted_price}")
            
            order_params = {
                'symbol': self.symbol, 
                'side': order_closing_side, 
                'positionSide': position_being_closed_side, 
                'type': 'LIMIT', 
                'timeInForce': 'GTC',
                'quantity': quantity, 
                'price': formatted_price
            }
            
            order = self.client.new_order(**order_params)
            self.logger.info(f"LIMIT CLOSE order sent. Response: {order}")
            return order

        except ClientError as e:
            self.logger.error(f"API ClientError placing LIMIT CLOSE for {self.symbol}: {e.error_message}")
        except Exception as e:
            self.logger.error(f"Unexpected error placing LIMIT CLOSE for {self.symbol}: {e}", exc_info=True)
        return None
    
    def _close_position_critically(self, reason: str):
        if not self.current_position_on_exchange or self.current_position_quantity <= 0:
            self.logger.info(f"Critical close called for {self.symbol}, but no active position to close. Reason: {reason}")
            if not self.bot_stopped_due_to_total_sl: 
                self.bot_stopped_due_to_total_sl = True
                self.logger.critical(f"Bot {self.symbol} stopped. Reason: {reason} (no active position found).")
            return

        self.logger.critical(f"CRITICAL ACTION for {self.symbol}: Attempting to close position {self.current_position_on_exchange} ({self.current_position_quantity}) by LIMIT due to: {reason}.")
        
        # --- AGGRESSIVE CLEANUP ---
        self.logger.warning(f"☢️ NUCLEAR OPTION: Cancelling ALL open orders before critical close.")
        self._cancel_all_orders_for_symbol()
        time.sleep(0.5) 

        close_order_side = "SELL" if self.current_position_on_exchange == "LONG" else "BUY"
        quantity_to_close = self.current_position_quantity 
        
        # Attempt 1: Aggressive Limit Close
        close_order = self._place_aggressive_limit_order_to_close(
            order_closing_side=close_order_side, 
            quantity=quantity_to_close, 
            position_being_closed_side=self.current_position_on_exchange
        )
        
        # Attempt 2: Retry if failed
        if not (close_order and close_order.get('orderId')):
             self.logger.error(f"First close attempt failed. Retrying...")
             time.sleep(0.5)
             close_order = self._place_aggressive_limit_order_to_close(
                order_closing_side=close_order_side, 
                quantity=quantity_to_close, 
                position_being_closed_side=self.current_position_on_exchange
            )

        if close_order and close_order.get('orderId'):
            self.logger.info(f"Limit close order for {self.symbol} sent due to '{reason}'. OrderID: {close_order['orderId']}. Bot will be stopped.")
        else:
            self.logger.error(f"FATAL for {self.symbol}: FAILED TO SEND CLOSE ORDER for '{reason}'. MANUAL INTERVENTION REQUIRED!")
        
        if not self.bot_stopped_due_to_total_sl: 
            self.bot_stopped_due_to_total_sl = True
            self.logger.critical(f"Bot {self.symbol} stopped due to critical error: {reason}.")


    def _load_model_performance(self) -> dict:
            """Loads statistics from JSON."""
            try:
                with open(self.performance_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
            except Exception as e:
                self.logger.error(f"Failed to load model statistics from {self.performance_file_path}: {e}", exc_info=True)
                return {} 
            
    def _save_model_performance(self, data: dict):
            """Saves statistics to JSON."""
            try:
                with open(self.performance_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                self.logger.error(f"CRITICAL SAVE ERROR: Failed to save model statistics to {self.performance_file_path}: {e}", exc_info=True)

    def _format_performance_report(self, performance_data: dict, symbol: str) -> str:
            """
            Formats report exactly as requested:
            Table with 'General' (Total) row at the end.
            """
            symbol_stats = performance_data.get(symbol, {})
            if not symbol_stats:
                return f"📊 **Report for {symbol}**: Statistics not yet collected."

            header = f"📊 **Accuracy Report for {symbol}**\n\n"
            table = "<pre>{:<20} | {:<10} | {:<5}\n".format("Model", "Win Rate", "%")
            table += "-"*42 + "\n"

            # Sort model names
            model_keys = sorted([key for key in symbol_stats.keys() if key != 'Ensemble'])
            for model_name in model_keys:
                stats = symbol_stats.get(model_name, {'correct': 0, 'total': 0})
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                win_rate = (correct / total * 100) if total > 0 else 0
                
                table += "{:<20} | {:<10} | {:<5.1f}\n".format(model_name, f"{correct}/{total}", win_rate)
            
            table += "-"*42 + "\n"

            # Ensemble / Total Row
            ensemble_stats = symbol_stats.get('Ensemble', {'correct': 0, 'total': 0})
            ens_correct = ensemble_stats.get('correct', 0)
            ens_total = ensemble_stats.get('total', 0)
            ens_win_rate = (ens_correct / ens_total * 100) if ens_total > 0 else 0
            
            table += "{:<20} | {:<10} | {:<5.1f}\n".format("Total", f"{ens_correct}/{ens_total}", ens_win_rate)

            return header + table + "</pre>"

    def _send_telegram_report(self, message: str):
        """Sends final HTML report to Telegram."""
        if self.telegram_bot_token_for_logging and self.telegram_chat_id_for_logging_str:
            try:
                chat_id = int(self.telegram_chat_id_for_logging_str)
                url = f"https://api.telegram.org/bot{self.telegram_bot_token_for_logging}/sendMessage"
                payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
                requests.post(url, data=payload, timeout=10)
                self.logger.info("Final performance report sent to Telegram.")
            except Exception as e:
                self.logger.error(f"Error sending performance report to Telegram: {e}")

    def run(self):
        """
        Main Thread Loop.
        FULL VERSION: With startup checks and GUARANTEED order cleanup.
        """
        self.logger.info(f"Attempting to start bot logic for {self.symbol}")
        shutdown_reason = "unexpected termination"
        self.bot_starting_virtual_capital = self.virtual_bot_capital_usdt
        self.current_position_on_exchange = None

        try:
            # ==================================================================
            # 1. INITIALIZATION & CHECKS
            # ==================================================================
            self._initialize_binance_client()
            
            # --- Check Positions ---
            self.logger.info(f"Checking existing open positions for {self.symbol} at startup...")
            existing_positions_list = self._get_open_positions(symbol_to_check=self.symbol)

            if existing_positions_list is None:
                self.logger.error(f"Failed to determine existing positions for {self.symbol}. Stopping.")
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"API error checking positions at startup for {self.symbol}"
            elif isinstance(existing_positions_list, list) and len(existing_positions_list) > 0:
                active_position_found = any(float(pos.get('positionAmt', 0)) != 0 for pos in existing_positions_list)
                if active_position_found:
                    pos_info_str = ", ".join([f"Qty: {p.get('positionAmt',0)}, Entry: {p.get('entryPrice',0)}" for p in existing_positions_list if float(p.get('positionAmt',0)) != 0])
                    self.logger.critical(f"!!! CRITICAL STARTUP WARNING for {self.symbol} !!!")
                    self.logger.critical(f"Found existing position: {pos_info_str}.")
                    self.logger.critical("Bot is designed to start FLAT. Bot will NOT start new trades.")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"active position found for {self.symbol} at startup"
            else: 
                self.logger.info(f"No existing open positions found for {self.symbol} at startup.")
            
            if self.should_stop_bot_flag_instance:
                self.logger.info(f"Bot for {self.symbol} stopped due to pre-flight checks: {shutdown_reason}")
                return

            # --- Trading Rules ---
            if not self._get_symbol_trading_rules():
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"failed to get trading rules for {self.symbol}"
                return
            
            # --- Leverage & Margin ---
            if not self._get_and_store_leverage_data():
                self.logger.warning(f"Failed to load leverage brackets for {self.symbol}. Using defaults.")
            
            if not self._set_isolated_margin():
                self.logger.warning(f"Failed to confirm isolated margin for {self.symbol}.")

            # --- Balance ---
            local_initial_balance_float = self._get_account_balance_usdt()
            if local_initial_balance_float is None:
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"failed to get account balance for {self.symbol}"
                return
            self.initial_account_balance_usdt = Decimal(str(local_initial_balance_float))

            if self.total_account_stop_loss_usdt > 0:
                self.stop_bot_threshold_balance_usdt = self.initial_account_balance_usdt - self.total_account_stop_loss_usdt
                self.logger.info(f"Initial Balance: {self.initial_account_balance_usdt:.2f} USDT. Global SL: {self.total_account_stop_loss_usdt:.2f} USDT. Stop Threshold: {self.stop_bot_threshold_balance_usdt:.2f} USDT for {self.symbol}.")
                if self.initial_account_balance_usdt < self.total_account_stop_loss_usdt:
                    self.logger.error(f"Initial balance ({self.initial_account_balance_usdt:.2f} USDT) is less than TOTAL_ACCOUNT_STOP_LOSS_USDT ({self.total_account_stop_loss_usdt:.2f} USDT). Cannot start {self.symbol}.")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"initial balance too low relative to global SL"
            else:
                self.stop_bot_threshold_balance_usdt = Decimal('0') 
                self.logger.info(f"Initial Balance: {self.initial_account_balance_usdt:.2f} USDT. TOTAL_ACCOUNT_STOP_LOSS_USDT not set or zero.")

            if self.should_stop_bot_flag_instance:
                self.logger.info(f"Bot for {self.symbol} stopped during initialization: {shutdown_reason}")
                return

            # --- Historical Data ---
            self.klines_df_main = self._get_klines_df_rest(self.interval, self.limit_main_tf)
            self.klines_df_higher_tf = self._get_klines_df_rest(self.selected_timeframe_higher, self.limit_higher_tf)

            if self.klines_df_main.empty or self.klines_df_higher_tf.empty:
                self.logger.error(f"Failed to load initial historical data for {self.symbol}. Stopping bot.")
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"failed to load historical data for {self.symbol}"
                return

            # --- Indicator Warmup ---
            try:
                df_calc = self._calculate_indicators(self.klines_df_main, self.klines_df_higher_tf)
                if df_calc.empty:
                    raise ValueError("Indicator calculation returned empty DataFrame.")
                self.logger.info("Initial indicators calculated successfully. System ready.")
            except Exception as e_init:
                self.logger.critical(f"Error checking indicators: {e_init}", exc_info=True)
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"indicator initialization error: {e_init}"
                return 

            # ==================================================================
            # 2. WEBSOCKET SETUP
            # ==================================================================
            self.last_processed_kline_open_time = self.klines_df_main.index[-1] if not self.klines_df_main.empty else None
            
            self.ws_client = UMFuturesWebsocketClient(
                on_message=self._on_websocket_message, 
                on_open=self._on_websocket_open,       
                on_close=self._on_websocket_close,     
                on_error=self._on_websocket_error      
            )
            
            self.listen_key = self._manage_user_data_stream()
            if not self.listen_key:
                self.logger.critical(f"Failed to obtain listen key for {self.symbol}. Stopping.")
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"failed to obtain listen key for {self.symbol}"
                return

            self.ws_client.user_data(listen_key=self.listen_key, id=0, callback=self._on_websocket_message)
            self.ws_client.kline(symbol=self.symbol, id=1, interval=self.interval, callback=self._on_websocket_message)
            self.ws_client.kline(symbol=self.symbol, id=2, interval=self.selected_timeframe_higher, callback=self._on_websocket_message)

            self.logger.info(f"Entering main operation loop for {self.symbol}.")
            
            websocket_restart_attempts = 0
            MAX_WEBSOCKET_RESTART_ATTEMPTS = self.config.get('MAX_WEBSOCKET_RESTART_ATTEMPTS', 3)

            # ==================================================================
            # 3. MAIN LOOP
            # ==================================================================
            while not self.should_stop_bot_flag_instance and not self.orchestrator_stop_event.is_set():
                current_time_main_loop = time.time()

                # Check flag files
                if os.path.exists(self.stop_flag_file_path_instance):
                    self.logger.critical(f"Stop flag {self.stop_flag_file_path_instance} found for {self.symbol}. Initiating stop...")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"command via flag file for {self.symbol}"
                    break 
                
                if self.orchestrator_stop_event.is_set():
                    self.logger.critical(f"Received global stop signal from orchestrator for {self.symbol}. Initiating stop...")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"global stop signal from orchestrator for {self.symbol}"
                    break
                
                # WS Restart Logic
                if self.reconnect_websocket_flag:
                    if websocket_restart_attempts < MAX_WEBSOCKET_RESTART_ATTEMPTS:
                        websocket_restart_attempts += 1
                        self.logger.warning(f"RECONNECT_WEBSOCKET_FLAG is True for {self.symbol}. Attempting WS re-init #{websocket_restart_attempts}/{MAX_WEBSOCKET_RESTART_ATTEMPTS}...")
                        self.reconnect_websocket_flag = False

                        if self.ws_client and hasattr(self.ws_client, 'stop') and callable(self.ws_client.stop):
                            try:
                                self.ws_client.stop()
                                time.sleep(3) 
                            except Exception as e_stop_ws:
                                self.logger.error(f"Error stopping ws_client during re-init: {e_stop_ws}")
                        
                        self.logger.info("Re-initializing WebSocket client and subscriptions...")
                        try:
                            self.ws_client = UMFuturesWebsocketClient(
                                on_message=self._on_websocket_message, on_open=self._on_websocket_open,
                                on_close=self._on_websocket_close, on_error=self._on_websocket_error
                            )
                            temp_listen_key = self._manage_user_data_stream(current_listen_key=self.listen_key)          
                            if temp_listen_key:
                                self.listen_key = temp_listen_key
                                self.ws_client.user_data(listen_key=self.listen_key, id=0, callback=self._on_websocket_message)
                                self.ws_client.kline(symbol=self.symbol, id=1, interval=self.interval, callback=self._on_websocket_message)
                                self.ws_client.kline(symbol=self.symbol, id=2, interval=self.selected_timeframe_higher, callback=self._on_websocket_message)
                                self.logger.info(f"WebSocket client for {self.symbol} successfully re-initialized.")
                            else: 
                                self.logger.critical(f"Failed to get listen key during WS re-init for {self.symbol}. Stopping.")
                                self.should_stop_bot_flag_instance = True
                                shutdown_reason = f"failed to get listen key during WS re-init for {self.symbol}"
                        except Exception as e_reconnect:
                            self.logger.error(f"ERROR during WS re-init: {e_reconnect}.", exc_info=True)
                            if websocket_restart_attempts >= MAX_WEBSOCKET_RESTART_ATTEMPTS:
                                self.should_stop_bot_flag_instance = True
                                shutdown_reason = f"max WS re-init attempts reached for {self.symbol}"
                    else: 
                        self.logger.critical(f"Exceeded max WS restart attempts for {self.symbol}. Stopping.")
                        self.should_stop_bot_flag_instance = True
                        shutdown_reason = f"max WS re-init attempts (flag not cleared) for {self.symbol}"
                    if self.should_stop_bot_flag_instance: break

                # Listen Key Renewal (every 30 mins)
                if not self.should_stop_bot_flag_instance and self.listen_key and \
                (self.last_listen_key_renewal_time == 0 or (current_time_main_loop - self.last_listen_key_renewal_time > 30 * 60)):
                    new_key_candidate = self._manage_user_data_stream(current_listen_key=self.listen_key)
                    if new_key_candidate:
                        if new_key_candidate != self.listen_key:
                            self.logger.warning(f"Listen key for {self.symbol} changed! Restarting WS.")
                            self.listen_key = new_key_candidate
                            self.reconnect_websocket_flag = True 
                            websocket_restart_attempts = 0
                    else: 
                        self.logger.critical(f"Failed to renew listen key for {self.symbol}. Stopping.")
                        self.should_stop_bot_flag_instance = True
                        shutdown_reason = f"failed to renew listen key for {self.symbol}"
                    if self.should_stop_bot_flag_instance: break

                time.sleep(1)
            
            # --- Stop Reason Refinement ---
            if self.orchestrator_stop_event.is_set() and shutdown_reason == "unexpected termination":
                shutdown_reason = f"global orchestrator stop for {self.symbol}"
            elif self.should_stop_bot_flag_instance and shutdown_reason == "unexpected termination":
                shutdown_reason = f"stop via individual flag for {self.symbol}"
            elif self.bot_stopped_due_to_total_sl and shutdown_reason == "unexpected termination":
                shutdown_reason = f"internal SL or critical error for {self.symbol}"
        
        except KeyboardInterrupt:
            self.logger.warning(f"KeyboardInterrupt received for {self.symbol}. Initiating stop...")
            self.should_stop_bot_flag_instance = True
            shutdown_reason = f"KeyboardInterrupt for {self.symbol}"
        except Exception as e_fatal:
            self.logger.critical(f"Fatal error in run method for {self.symbol}: {e_fatal}", exc_info=True)
            self.should_stop_bot_flag_instance = True
            shutdown_reason = f"fatal error in {self.symbol}: {str(e_fatal)[:200]}"
        
        finally:
            # ==================================================================
            # 4. GUARANTEED SHUTDOWN
            # ==================================================================
            self.logger.info(f"Starting shutdown sequence for {self.symbol} (Reason: {shutdown_reason})...")
            
            # 1. GUARANTEED ORDER REMOVAL
            self.logger.info(f"🧹 CLEANUP: Removing ALL orders for {self.symbol}...")
            # First attempt
            self._cancel_all_orders_for_symbol()
            time.sleep(0.5)
            # Second attempt (control)
            self._cancel_all_orders_for_symbol()

            # ==================================================================
            # 🔥 EMERGENCY POSITION CLOSE ON STOP 🔥
            # ==================================================================
            # Check internal bot state for active position
            if self.current_position_on_exchange and self.current_position_quantity > 0:
                self.logger.warning(f"🚨 EMERGENCY STOP: Closing open position {self.current_position_on_exchange} {self.current_position_quantity} {self.symbol}...")
                
                # Determine close side
                side_to_close = 'SELL' if self.current_position_on_exchange == 'LONG' else 'BUY'
                
                try:
                    # Use existing method to form Market Order
                    # Args: (side, quantity, position_side)
                    self._place_market_order_to_close(side_to_close, self.current_position_quantity, self.current_position_on_exchange)
                    self.logger.info("✅ EMERGENCY: Market Close Order sent successfully during shutdown.")
                    # Give 2 seconds for execution before killing process
                    time.sleep(2.0)
                except Exception as e:
                    self.logger.error(f"❌ CRITICAL: Failed to close position during stop: {e}")
            else:
                self.logger.info(f"Stop Check: No internal open position detected for {self.symbol}.")
            # ==================================================================

            # 3. Stop WebSocket
            if self.ws_client and hasattr(self.ws_client, 'stop') and callable(self.ws_client.stop):
                self.logger.info(f"Closing WebSocket for {self.symbol}...")
                try:
                    self.ws_client.stop() 
                except Exception as e_stop: 
                    self.logger.error(f"Error stopping WS: {e_stop}")
            
            # 4. Remove Flags
            if os.path.exists(self.stop_flag_file_path_instance):
                try: os.remove(self.stop_flag_file_path_instance)
                except: pass

            # 5. Final Message
            final_bot_status_message = f"🤖 Trading bot for *{self.symbol}* has stopped.\n🏁 *Reason*: `{shutdown_reason}`."
            self._send_final_telegram_message(final_bot_status_message)
            
            self.logger.info(f"Trading bot for {self.symbol} shut down safely.")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] Bot {self.symbol} shutdown complete.")

def prepare_for_backtest(self):
    print("Preparing bot instance for backtesting...")
    try:
        # Initialize client and load trading rules
        self.client = UMFutures()
        
        # Just call the method. It populates self.symbol_trading_rules internally.
        # We check if it returned True (success).
        if not self._get_symbol_trading_rules():
            # If method returned False, an error occurred
            raise ValueError("Failed to load trading rules during backtest initialization.")
        
        # At this point self.symbol_trading_rules is a valid dictionary.
        
        self.strategy_params = self.config.get('strategy_params', {})
        self.n1 = self.strategy_params.get('n1', 20)
        self.n2 = self.strategy_params.get('n2', 10)
        self.atr_multiplier = self.strategy_params.get('atr_multiplier', 2.0)
        self.entry_retracement_pct = Decimal(str(self.strategy_params.get('entry_retracement_pct', '0.5')))
        self.tp_multiplier = Decimal(str(self.strategy_params.get('TP_MULTIPLIER', '1.0')))
        self.sl_multiplier = Decimal(str(self.strategy_params.get('SL_MULTIPLIER', '0.2')))

        print("Backtest initialization successful.")
    except Exception as e:
        print(f"Error during special backtest initialization: {e}")
        raise

    
    def setup_for_backtest(self):
        """Prepares bot instance for backtesting."""
        print("Setting up bot instance for backtest...")
        # Explicitly set parameters from config
        strategy_params = self.config.get('strategy_params', {})
        self.n1 = strategy_params.get('n1', 20)
        self.n2 = strategy_params.get('n2', 10)
        self.atr_multiplier = strategy_params.get('atr_multiplier', 2.0)
        self.entry_retracement_pct = Decimal(str(strategy_params.get('entry_retracement_pct', '0.5')))
        self.tp_multiplier = Decimal(str(strategy_params.get('TP_MULTIPLIER', '1.0')))
        self.sl_multiplier = Decimal(str(strategy_params.get('SL_MULTIPLIER', '0.2')))
        print(f"Backtest parameters set: n1={self.n1}, SL_Multiplier={self.sl_multiplier}")

    

# --- Orchestrator Class ---
class Orchestrator:
    def __init__(self, config_file_path: str):
        self.config_file_path = config_file_path
        self.active_bot_instances = {}  # symbol: TradingBotInstance
        self.active_bot_threads = {}    # symbol: threading.Thread
        self.last_config_mtime = 0
        self.global_stop_event = threading.Event()

        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN") 
        self.telegram_chat_id_str = os.getenv("YOUR_TELEGRAM_CHAT_ID") 

        # Orchestrator Logger Setup
        self.logger = logging.getLogger("Orchestrator")
        if not self.logger.handlers: # Avoid duplication if script is run multiple times in one env
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d UTC %(levelname)-8s %(name)-15s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # File logging can be enabled here if needed
            # log_file_orchestrator = os.path.join(SCRIPT_DIR_NEW_LOGIC, "orchestrator_main.log")
            # fh = logging.FileHandler(log_file_orchestrator, mode='a', encoding='utf-8')
            # fh.setFormatter(formatter)
            # self.logger.addHandler(fh)

        if not all([self.api_key, self.api_secret]):
            self.logger.critical("Binance API_KEY or API_SECRET not found. Orchestrator cannot start.")
            raise ValueError("Binance API keys not found for Orchestrator.")
        
        self.logger.info("Orchestrator initialized.")

    def _load_raw_config_from_file(self) -> list:
        self.logger.debug(f"Attempting to load configuration from {self.config_file_path}")
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                loaded_configs = json.load(f)
            if not isinstance(loaded_configs, list):
                self.logger.error(f"Configuration in {self.config_file_path} is not a list. Returning empty list.")
                return []
            self.logger.info(f"Successfully loaded {len(loaded_configs)} raw configurations from {self.config_file_path}.")
            return loaded_configs
        except FileNotFoundError:
            self.logger.warning(f"Configuration file {self.config_file_path} not found. Returning empty list.")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {self.config_file_path}: {e}. Returning empty list.")
            return []
        except Exception as e:
            self.logger.error(f"Failed to load configurations from {self.config_file_path}: {e}", exc_info=True)
            return []

    def _prepare_bots_configurations(self, raw_configs: list) -> list:
        if not raw_configs:
            return []

        # Calculate Global SL based on file capital
        total_initial_capital_sum = sum(
            Decimal(str(item.get("config", {}).get('INITIAL_CAPITAL_PER_TRADE_USDT', '0.0')))
            for item in raw_configs
        )
        global_total_account_sl_value = (total_initial_capital_sum / Decimal('3.0')) if total_initial_capital_sum > 0 else Decimal('0.0')

        processed_configurations = []
        for bot_conf_item in raw_configs:
            symbol = bot_conf_item["symbol"]
            config_from_file = bot_conf_item.get("config", {})
            
            # If bot is already running, use its current capital, not initial
            if symbol in self.active_bot_instances:
                current_instance_capital = self.active_bot_instances[symbol].virtual_bot_capital_usdt
                config_from_file['VIRTUAL_BOT_CAPITAL_USDT'] = str(current_instance_capital)
                self.logger.info(f"Preserving existing capital for active bot {symbol}: {current_instance_capital:.4f} USDT")

            # Add calculated Global SL
            config_from_file['TOTAL_ACCOUNT_STOP_LOSS_USDT'] = str(global_total_account_sl_value)

            processed_configurations.append({
                "symbol": symbol,
                "interval": config_from_file.get("interval", "5m"),
                "config": config_from_file
            })
        self.logger.info(f"Prepared {len(processed_configurations)} full bot configurations.")
        return processed_configurations

    def _start_bot_instance(self, bot_params: dict):
        symbol = bot_params["symbol"]
        if symbol in self.active_bot_instances:
            self.logger.warning(f"Bot for {symbol} is already running. Skipping start.")
            return

        self.logger.info(f"Attempting to start bot for {symbol} with interval {bot_params.get('interval', 'N/A')}")
        
        individual_stop_flag = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"STOP_TRADING_BOT_{symbol}.flag")
        if os.path.exists(individual_stop_flag):
            try:
                os.remove(individual_stop_flag)
                self.logger.info(f"Removed old individual stop flag for {symbol}: {individual_stop_flag}")
            except OSError as e:
                self.logger.error(f"Failed to remove old individual stop flag for {symbol}: {e}")

        try:
            instance = TradingBotInstance(
                symbol=symbol,
                interval=bot_params["interval"],
                api_key=self.api_key,
                api_secret=self.api_secret,
                telegram_bot_token=self.telegram_bot_token,
                telegram_chat_id_str=self.telegram_chat_id_str,
                bot_config=bot_params["config"],
                orchestrator_stop_event=self.global_stop_event
            )
        except Exception as e_init:
            self.logger.error(f"Error initializing TradingBotInstance for {symbol}: {e_init}", exc_info=True)
            return # Do not start thread if init failed
        
        # --- DIAGNOSTIC BLOCK ---
        print("\n" + "="*20 + f" OBJECT DIAGNOSTICS FOR {symbol} " + "="*20)
        try:
            print(f"OBJECT TYPE: {type(instance)}")
            print(f"Has attribute 'run'? -> {hasattr(instance, 'run')}")
            print("AVAILABLE ATTRIBUTES AND METHODS:")
            # Print methods that do not start with '__'
            print([attr for attr in dir(instance) if not attr.startswith('__')])
        except Exception as e_diag:
            print(f"Error during diagnostics: {e_diag}")
        print("="*60 + "\n")
        # --- END DIAGNOSTIC BLOCK ---

          
        thread = threading.Thread(target=instance.run, name=f"BotThread-{symbol}")
        self.active_bot_instances[symbol] = instance
        self.active_bot_threads[symbol] = thread
        thread.start()
        self.logger.info(f"Bot for {symbol} successfully started in thread {thread.name}.")

    def _stop_bot_instance(self, symbol: str, reason: str = "config_update"):
        if symbol not in self.active_bot_instances:
            self.logger.debug(f"Bot for {symbol} not found for stopping (might be already stopped).")
            return

        self.logger.info(f"Initiating stop for {symbol}. Reason: {reason}.")
        instance = self.active_bot_instances.get(symbol)
        thread = self.active_bot_threads.get(symbol)

        if instance:
            instance.should_stop_bot_flag_instance = True
        
        if thread:
            self.logger.info(f"Waiting for thread {thread.name} for {symbol} to finish (timeout 60s)...")
            thread.join(timeout=60)
            if thread.is_alive():
                self.logger.warning(f"Thread {thread.name} for {symbol} did not finish correctly after timeout.")
            else:
                self.logger.info(f"Thread {thread.name} for {symbol} finished successfully.")
        
        # Remove from active lists regardless of clean thread exit
        if symbol in self.active_bot_instances: del self.active_bot_instances[symbol]
        if symbol in self.active_bot_threads: del self.active_bot_threads[symbol]
        self.logger.info(f"Bot instance for {symbol} removed from Orchestrator active lists.")


    def _load_and_reconcile_bots(self):
        self.logger.info(f"Checking for changes and reconciling bots per {self.config_file_path}...")
        
        full_new_bot_configurations = self._load_and_prepare_configs()

        new_config_symbols = set(bot_conf["symbol"] for bot_conf in full_new_bot_configurations)
        current_running_symbols = set(self.active_bot_instances.keys())

        # --- STEP 1: Stop bots removed from config ---
        symbols_to_stop = current_running_symbols - new_config_symbols
        if symbols_to_stop:
            self.logger.info(f"Symbols to stop (removed from config): {symbols_to_stop}")
            for symbol in symbols_to_stop:
                self._stop_bot_instance(symbol, reason="Removed from configuration")

        # --- STEP 2: Start new bots ---
        symbols_to_start = new_config_symbols - current_running_symbols
        if symbols_to_start:
            self.logger.info(f"Symbols to start (new in config): {symbols_to_start}")
            for symbol_to_add in symbols_to_start:
                # Find full config for new bot
                bot_params = next((conf for conf in full_new_bot_configurations if conf["symbol"] == symbol_to_add), None)
                if bot_params:
                    # (USE_ML_FILTER check REMOVED)
                    
                    time.sleep(2) # Small delay between starts
                    self._start_bot_instance(bot_params)
                else:
                    self.logger.error(f"Could not find configuration for new symbol {symbol_to_add}.")

        # --- STEP 3: Ignore running bots that haven't changed ---
        symbols_to_keep = current_running_symbols.intersection(new_config_symbols)
        if symbols_to_keep:
            self.logger.info(f"Symbols continuing operation without changes: {symbols_to_keep}")

        # Update last modification time
        if os.path.exists(self.config_file_path):
            self.last_config_mtime = os.path.getmtime(self.config_file_path)
        else:
            self.last_config_mtime = 0

        self.logger.info("Bot reconciliation completed.")

    def _load_and_prepare_configs(self):
        """Helper to load and prepare configurations."""
        raw_configs = self._load_raw_config_from_file()
        return self._prepare_bots_configurations(raw_configs)

    def run_main_loop(self):
        self.logger.info("Orchestrator started. Entering main monitoring loop...")
        
        # Initial load and start
        if os.path.exists(self.config_file_path):
            try:
                self.last_config_mtime = os.path.getmtime(self.config_file_path)
            except OSError:
                self.logger.error(f"Failed to get modification time for {self.config_file_path} at start.")
                self.last_config_mtime = 0
            self._load_and_reconcile_bots()
        else:
            self.logger.warning(f"Initial config file {self.config_file_path} not found. Orchestrator will wait for creation or RELOAD flag.")
            self.last_config_mtime = 0

        try:
            while not self.global_stop_event.is_set():
                if os.path.exists(GLOBAL_STOP_FLAG_FILE_PATH):
                    self.logger.critical(f"!!! {GLOBAL_STOP_FLAG_FILE_PATH} detected! Initiating Orchestrator shutdown.")
                    self.global_stop_event.set()
                    try:
                        os.remove(GLOBAL_STOP_FLAG_FILE_PATH)
                        self.logger.info(f"File {GLOBAL_STOP_FLAG_FILE_PATH} removed by Orchestrator.")
                    except OSError as e:
                        self.logger.error(f"Error removing {GLOBAL_STOP_FLAG_FILE_PATH}: {e}")
                    break 

                if os.path.exists(RELOAD_CONFIG_FLAG_FILE_PATH):
                    self.logger.info(f"Flag {RELOAD_CONFIG_FLAG_FILE_PATH} detected. Reloading configuration.")
                    try:
                        os.remove(RELOAD_CONFIG_FLAG_FILE_PATH)
                        self.logger.info(f"File {RELOAD_CONFIG_FLAG_FILE_PATH} removed.")
                    except OSError as e:
                        self.logger.error(f"Error removing {RELOAD_CONFIG_FLAG_FILE_PATH}: {e}")
                    
                    self._load_and_reconcile_bots() # Run reconciliation
                else: # If no flag, check file mtime (alternative)
                    if os.path.exists(self.config_file_path):
                        try:
                            current_mtime = os.path.getmtime(self.config_file_path)
                            if current_mtime != self.last_config_mtime:
                                self.logger.info(f"Detected change in {self.config_file_path} via modification time.")
                                self._load_and_reconcile_bots()
                            # self.last_config_mtime updated inside _load_and_reconcile_bots
                        except OSError:
                            self.logger.debug(f"Failed to check modification time for {self.config_file_path}.")
                    elif self.last_config_mtime != 0 : # File was there, now gone
                        self.logger.warning(f"Config file {self.config_file_path} was deleted. Stopping all bots.")
                        self._load_and_reconcile_bots() # This will stop all bots since config is empty


                # Check thread health
                for symbol, thread in list(self.active_bot_threads.items()):
                    if not thread.is_alive():
                        self.logger.warning(f"Thread for {symbol} ({thread.name}) terminated unexpectedly.")
                        if symbol in self.active_bot_instances: del self.active_bot_instances[symbol]
                        del self.active_bot_threads[symbol]
                        # Auto-restart logic can be added here if needed

                time.sleep(10) # Check interval

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received by Orchestrator. Initiating shutdown.")
            self.global_stop_event.set()
        except Exception as e_main_loop:
            self.logger.critical(f"Unhandled error in Orchestrator main loop: {e_main_loop}", exc_info=True)
            self.global_stop_event.set() # Attempt graceful stop
        finally:
            self.logger.info("Starting final Orchestrator shutdown procedure...")
            if not self.global_stop_event.is_set():
                self.logger.info("Setting global_stop_event in finally block.")
                self.global_stop_event.set()

            active_symbols_at_shutdown = list(self.active_bot_instances.keys())
            if active_symbols_at_shutdown:
                self.logger.info(f"Stopping remaining active bots: {active_symbols_at_shutdown}")
                for symbol in active_symbols_at_shutdown:
                    self._stop_bot_instance(symbol, reason="orchestrator_global_shutdown")
            
            # Extra wait for threads
            final_threads_to_join = [t for t in self.active_bot_threads.values() if t.is_alive()]
            if final_threads_to_join:
                 self.logger.info(f"Waiting for {len(final_threads_to_join)} threads to finish...")
                 for thread in final_threads_to_join:
                     thread.join(timeout=15)
                     if thread.is_alive():
                         self.logger.warning(f"Thread {thread.name} still active after final join.")
            
            # Clean up flags
            if os.path.exists(GLOBAL_STOP_FLAG_FILE_PATH):
                try:
                    os.remove(GLOBAL_STOP_FLAG_FILE_PATH)
                    self.logger.info(f"File {GLOBAL_STOP_FLAG_FILE_PATH} removed by Orchestrator on exit.")
                except OSError: pass
            if os.path.exists(RELOAD_CONFIG_FLAG_FILE_PATH):
                try:
                    os.remove(RELOAD_CONFIG_FLAG_FILE_PATH)
                except OSError: pass
            
            self.logger.info("Orchestrator has shut down completely.")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    # UTF-8 Setup for Windows Console
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except: pass
            
    parser = argparse.ArgumentParser(description="High-Frequency Statistical Bot (High/Low Logic)")
    
    # Run Modes
    group = parser.add_mutually_exclusive_group(required=True) 
    group.add_argument("--run-single", type=str, metavar="SYMBOL", help="Run SINGLE bot instance (Live Trading).")
    group.add_argument("--run-orchestrator", action='store_true', help="Run ORCHESTRATOR (Group Management).")
    
    # Extra Args
    parser.add_argument("--config", type=str, default=DEFAULT_BOT_CONFIG_FILE, help="Path to config file (for Orchestrator).")
    parser.add_argument("--no-progress-bar", action='store_true', help="Disable tqdm (clean logs).")
    
    args = parser.parse_args()

    # --- RUN SINGLE BOT ---
    if args.run_single:
        # Load Keys
        main_logger.info("Loading .env keys...")
        load_dotenv(dotenv_path=KEY_ENV_PATH_LOGIC)
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        tg_id = os.getenv('YOUR_TELEGRAM_CHAT_ID')

        if not api_key or not api_secret:
            main_logger.critical("❌ Binance API Keys not found! Check key.env")
            sys.exit(1)

        symbol = args.run_single.upper()
        main_logger.info(f"--- STARTING BOT: {symbol} ---")
        
        # Try load config from file
        bot_config = {}
        if os.path.exists(args.config):
            try:
                with open(args.config, 'r', encoding='utf-8') as f:
                    full_conf = json.load(f)
                    # Find config for this symbol
                    found = next((i for i in full_conf if i["symbol"] == symbol), None)
                    if found: bot_config = found.get("config", {})
            except: pass

        # Create and run instance
        bot = TradingBotInstance(
            symbol=symbol,
            interval=bot_config.get("interval", "5m"), # Default 5m
            api_key=api_key,
            api_secret=api_secret,
            bot_config=bot_config,
            telegram_bot_token=tg_token,
            telegram_chat_id_str=tg_id
        )
        bot.run()

    # --- RUN ORCHESTRATOR ---
    elif args.run_orchestrator:
        main_logger.info("--- STARTING ORCHESTRATOR ---")
        
        if not args.config or not os.path.exists(args.config):
            main_logger.critical(f"❌ Configuration file {args.config} not found!")
            sys.exit(1)
            
        orchestrator = Orchestrator(config_file_path=args.config)
        orchestrator.run_main_loop()
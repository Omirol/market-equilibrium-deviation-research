# --- БЛОК 1: ЗАГАЛЬНІ ІМПОРТИ (з обох файлів) ---
import os
import time
import logging
import sys
import requests
import json
import argparse
import numpy as np
import pandas as pd
import numba
from typing import Tuple
from dotenv import load_dotenv
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.error import ClientError
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
from numba import jit
import threading
import pandas_ta as ta
import joblib
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report, f1_score, precision_recall_curve, recall_score
from catboost import CatBoostClassifier
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import subprocess
import warnings # Додайте цей імпорт
from collections import deque
import tensorflow as tf
# --- Оновлені імпорти для Keras / Transformer ---
from keras.models import Model
from keras import layers
from sklearn.tree import DecisionTreeClassifier
# ЗМІНА: Додано LSTM
from keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D, Add, Embedding, Concatenate, Flatten, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
# --- НОВІ ІМПОРТИ ДЛЯ РОЗШИРЕНОГО НАБОРУ МОДЕЛЕЙ ---
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# ---------------------------------------------------
# ЗМІНА: Додано нові класифікатори
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# ---------------------------------------------------
# --- ЗМІНА: Додаємо нові моделі ---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
# ---------------------------------------------------

class PositionalEncoding(layers.Layer):
    """
    ФІНАЛЬНА, ПОВНІСТЮ СУМІСНА ВЕРСІЯ.
    Конструктор тепер приймає стандартні Keras аргументи (**kwargs).
    """
    def __init__(self, position, d_model, **kwargs): # <-- ЗМІНА 1: Додано **kwargs
        super(PositionalEncoding, self).__init__(**kwargs) # <-- ЗМІНА 2: Передаємо kwargs батьківському класу
        self.pos_encoding = self.positional_encoding(position, d_model)

    # get_config потрібен для коректного збереження/завантаження
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        # Ми не зберігаємо self.pos_encoding, оскільки він розраховується в __init__
        # Зберігаємо тільки ті параметри, які потрібні для його відтворення
        config.update({
            'position': self.pos_encoding.shape[1],
            'd_model': self.pos_encoding.shape[2],
        })
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def focal_loss(gamma=2.0):
    """
    ФІНАЛЬНА, НАДІЙНА ВЕРСІЯ. Повністю переписана на чистому TensorFlow,
    щоб уникнути будь-яких проблем з `keras.backend`.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Переконуємось, що y_true має тип float32
        y_true = tf.cast(y_true, tf.float32)
        
        # Розраховуємо pt - ймовірність для правильного класу
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Розраховуємо loss, використовуючи тільки функції з `tf`
        loss = -tf.pow(1. - pt, gamma) * tf.math.log(pt + tf.keras.backend.epsilon())
        
        # Усереднюємо loss по всьому батчу
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# --- НАЛАШТУВАННЯ ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
getcontext().prec = 12

# !!! ВАЖЛИВО: "СЛОВНИК" РИНКУ (Токенізація) !!!
# Це правила перекладу цифр у "слова" для Трансформера
MARKET_VOCAB = {
    # Speed: 0=Stall (стояк), 1=Grind (повзе), 2=Impulse (летить)
    'speed_bins': [-np.inf, 0.5, 2.0, np.inf], 
    # Volume: 0=Low, 1=Normal, 2=High (зусилля)
    'vol_bins':   [-np.inf, 0.8, 2.5, np.inf], 
    # Zone: 0=Під рівнем, 1=НА РІВНІ (Touch), 2=Над рівнем
    'zone_bins':  [-np.inf, -0.15, 0.15, np.inf] 
}

# !!! ВАЖЛИВО: Жорстка фіксація шляху до папки скрипта !!!
SCRIPT_DIR_NEW_LOGIC = os.path.dirname(os.path.abspath(__file__))
KEY_ENV_PATH_LOGIC = os.path.join(SCRIPT_DIR_NEW_LOGIC, "key.env")
load_dotenv(dotenv_path=KEY_ENV_PATH_LOGIC)

GLOBAL_STOP_FLAG_FILE_PATH = os.path.join(SCRIPT_DIR_NEW_LOGIC, "STOP_BOT_NOW.flag")
RELOAD_CONFIG_FLAG_FILE_PATH = os.path.join(SCRIPT_DIR_NEW_LOGIC, "RELOAD_CONFIG.flag")
DEFAULT_BOT_CONFIG_FILE = os.path.join(SCRIPT_DIR_NEW_LOGIC, "active_bots_config.json")

# --- БЛОК 2.1: МАТРИЧНА ФІЗИКА (CONSTANTS & NUMBA) ---

FULL_STEP_GRID = np.array([
    0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 
    25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0
], dtype=np.float64)

@numba.jit(nopython=True, cache=True)
def calculate_dynamic_physics_numba(closes, volumes, active_steps, atrs):
    """
    Рахує фізику (V, E, R) динамічно для кожного рядка, 
    використовуючи індивідуальний active_step для цього моменту часу.
    """
    n = len(closes)
    
    # Вихідні масиви (Фізика)
    velocity = np.zeros(n, dtype=np.float64) # V: Швидкість пробою
    effort = np.zeros(n, dtype=np.float64)   # E: Зусилля (Об'єм / SMA)
    result = np.zeros(n, dtype=np.float64)   # R: Результат (Рух ціни / ATR)
    
    # Геометрія (Де ми?)
    dist_to_level = np.zeros(n, dtype=np.float64) # Відстань до найближчого рівня
    
    for i in range(1, n):
        price = closes[i]
        step = active_steps[i] # Беремо крок, актуальний САМЕ ЗАРАЗ
        atr = atrs[i]
        vol = volumes[i]
        
        if step == 0 or atr == 0: continue

        # 1. ГЕОМЕТРІЯ: Знаходимо найближчий рівень сітки
        nearest_level = round(price / step) * step
        # Нормалізована відстань (-1..0..1, де 0 - це рівень)
        dist_to_level[i] = (price - nearest_level) / step

        # 2. ФІЗИКА: Result (Рух ціни в ATR)
        # Наскільки ціна змінилася відносно волатильності
        change = abs(closes[i] - closes[i-1])
        result[i] = change / atr
        
        # 3. ФІЗИКА: Velocity (Швидкість)
        # Якщо ми пройшли велику відстань за 1 свічку - це імпульс
        velocity[i] = change / step
        
        # 4. ФІЗИКА: Effort (Зусилля - тут поки просто сирий, нормалізація буде в pandas)
        effort[i] = vol 

    return velocity, effort, result, dist_to_level

@numba.jit(nopython=True, cache=True)
def calculate_single_step_physics_numba(closes, step_val):
    """
    Рахує Velocity та Distance для одного фіксованого кроку (step_val)
    по всьому масиву цін.
    """
    n = len(closes)
    velocity = np.zeros(n, dtype=np.float64)
    dist_to_level = np.zeros(n, dtype=np.float64)
    
    # Якщо крок 0 або NaN - повертаємо нулі
    if step_val <= 0 or np.isnan(step_val):
        return velocity, dist_to_level

    for i in range(1, n):
        price = closes[i]
        
        # 1. ГЕОМЕТРІЯ (Distance)
        nearest_level = round(price / step_val) * step_val
        dist_to_level[i] = (price - nearest_level) / step_val

        # 2. ФІЗИКА (Velocity)
        change = abs(closes[i] - closes[i-1])
        velocity[i] = change / step_val
        
    return velocity, dist_to_level

# Глобальні константи
# Глобальні константи
TIMESTEPS = 48

# Ми маємо 9 сіток (0..8)
NUM_GRIDS = 9 

# 1. Формуємо список колонок-токенів автоматично
# Буде: ['t_zone_0', 't_speed_0', 't_zone_1', 't_speed_1', ... , 't_vol']
TOKEN_FEATURES = []
for i in range(NUM_GRIDS):
    TOKEN_FEATURES.append(f't_zone_{i}')
    TOKEN_FEATURES.append(f't_speed_{i}')

# Додаємо глобальний токен об'єму (він один)
TOKEN_FEATURES.append('t_vol') 

# 2. Числові фічі (Global Floats)
# Можна додати сюди і dist_i, vel_i, якщо хочеш подавати і токени, і числа.
# Але для початку подамо тільки глобальну фізику + мульти-токени.
FLOAT_FEATURES = ['result', 'effort', 'absorption'] 

# Загальна кількість
NUM_FLOAT_FEATURES = len(FLOAT_FEATURES)

# !!! ВИПРАВЛЕНО: СУФІКСИ ЗАМІСТЬ ФІКСОВАНИХ ІМЕН !!!
# Повні імена будуть: BTCUSDT_StepPredictor_Scaler_Floats.joblib
SCALER_FILE_SUFFIX = "_StepPredictor_Scaler_Floats.joblib"
MODEL_FILE_SUFFIX = "_Hybrid_Transformer_Model.keras"

# Логгер
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC %(levelname)-8s %(name)-20s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
main_logger = logging.getLogger(__name__)

def get_all_historical_futures_data(symbol, interval, start_str=None, end_str=None, lookback_months=None):
    """
    ЗАВАНТАЖУЄ ІСТОРІЮ за вказаний період або за останні N місяців.
    Параметр lookback_months має пріоритет.
    
    ★★★ ВЕРСІЯ, ВИПРАВЛЕНА ДЛЯ ВАЛІДАЦІЇ 1-В-1 ЯК РЕАЛЬНИЙ БОТ ★★★
    1.  Прибирає 'utc=True' (1-в-1 як у бота).
    2.  Додає 'keep=last' в drop_duplicates (1-в-1 як у бота).
    3.  !!! ДОДАНО ОБРІЗАННЯ КОЛОНОК (1-в-1 як у бота) !!!
    """
    load_dotenv(dotenv_path="key.env")
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = UMFutures(key=api_key, secret=api_secret)

    if lookback_months and lookback_months > 0:
        end_dt_calc = datetime.now()
        start_dt_calc = end_dt_calc - timedelta(days=int(lookback_months * 30.5))
        start_str = start_dt_calc.strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"НАВЧАННЯ: Використовується lookback_months={lookback_months}. Розрахована дата старту: {start_str}")
    elif not start_str:
        raise ValueError("Необхідно вказати або start_str, або lookback_months")

    logging.info(f"НАВЧАННЯ: Завантаження повної історії: {symbol} ({interval}) з {start_str} до {end_str or 'поточного часу'}...")
    start_ts = int(datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000) if not end_str else int(datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

    all_klines = []
    current_start_ts = start_ts

    try:
        disable_progress = args.no_progress_bar
    except NameError:
        disable_progress = False

    with tqdm(total=(end_ts - start_ts), unit='ms', desc=f"Завантаження {symbol}", disable=disable_progress) as pbar:
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
    
    # --- ★★★ ОСТАТОЧНЕ ВИПРАВЛЕННЯ (1-в-1 ЯК У БОТА) ★★★ ---
    # Обрізаємо DF до 5 колонок + timestamp, ЯК ЦЕ РОБИТЬ _get_klines_df_rest
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    # --- ★★★ КІНЕЦЬ ВИПРАВЛЕННЯ ★★★ ---

    # --- (Ці виправлення в тебе вже мають бути з минулого разу) ---
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str) if end_str else pd.to_datetime(datetime.now())
    # --- (Кінець старих виправлень) ---

    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]

    # Цей цикл 'for' тепер не потрібен, бо 'astype(float)' вже все зробив,
    # але хай залишається, він не шкодить.
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('timestamp', inplace=True)
    return df



def build_hybrid_transformer_model(timesteps, num_float_features):
    """
    ГІБРИДНА МОДЕЛЬ:
    Вхід 1 (Категорії): [Zone, Speed, Volume] -> Embedding
    Вхід 2 (Числа): [Dist, Effort, Absorption] -> Dense
    """
    # --- ВХІД 1: ТОКЕНИ (СЛОВА) ---
    # Розмірності: (Batch, Timesteps, 1) для кожного токена
    in_zone = Input(shape=(timesteps,), name='in_zone')
    in_speed = Input(shape=(timesteps,), name='in_speed')
    in_vol = Input(shape=(timesteps,), name='in_vol')
    
    # Embedding шари (Вчать "сенс" кожного стану)
    # 3 категорії зон -> вектор розміром 4
    emb_zone = Embedding(input_dim=4, output_dim=4)(in_zone) 
    emb_speed = Embedding(input_dim=4, output_dim=4)(in_speed)
    emb_vol = Embedding(input_dim=4, output_dim=4)(in_vol)
    
    # Об'єднуємо ембеддінги
    # Shape: (Batch, Timesteps, 12)
    merged_tokens = Concatenate()([emb_zone, emb_speed, emb_vol])
    
    # --- ВХІД 2: ЧИСЛА (ФАКТИ) ---
    in_floats = Input(shape=(timesteps, num_float_features), name='in_floats')
    # Трохи обробляємо числа перед злиттям
    x_floats = layers.Dense(16, activation='relu')(in_floats)
    
    # --- ЗЛИТТЯ ВСЬОГО ---
    # Shape: (Batch, Timesteps, 12+16=28)
    x = Concatenate()([merged_tokens, x_floats])
    
    # --- TRANSFORMER BLOCK (Як і було, але тепер дані "розумні") ---
    # Positional Encoding додаємо тут
    x = PositionalEncoding(position=timesteps, d_model=28)(x)
    
    for _ in range(2): # 2 блоки трансформера
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=4)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        ffn = layers.Dense(32, activation="relu")(x)
        ffn = layers.Dense(28)(ffn) # Повертаємо до розмірності d_model
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
    # --- ГОЛОВА (HEAD) ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[in_zone, in_speed, in_vol, in_floats], outputs=outputs)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', # Або focal_loss()
                  metrics=['accuracy'])
    
    return model


# --- ПРАВИЛЬНА ВЕРСІЯ ФУНКЦІЇ ---
def create_sequences(full_feature_df: pd.DataFrame, labels_series: pd.Series, timesteps: int, full_df_indices: dict):
    """
    ОПТИМІЗОВАНА версія.
    Правильно створює послідовності, використовуючи ПЕРЕДАНИЙ словник індексів.
    """
    X_seq, y_seq = [], []
    
    for signal_timestamp, label in labels_series.items():
        signal_idx = full_df_indices.get(signal_timestamp)
        
        # Переконуємось, що сигнал є в основному датафреймі і що перед ним є достатньо даних
        if signal_idx is not None and signal_idx >= timesteps:
            # Вирізаємо послідовність з timesteps свічок, що передують сигналу
            sequence = full_feature_df.iloc[signal_idx - timesteps : signal_idx]
            X_seq.append(sequence.values)
            y_seq.append(label)
            
    return np.array(X_seq), np.array(y_seq)



def calculate_dynamic_lookahead(df, coefficient=1.5):
    """
    Розраховує динамічний горизонт для бек-тесту на основі макс. довжини кроку.
    """
    max_step_duration = df['step_duration_candles'].max()
    lookahead = int(max_step_duration * coefficient)
    logging.info(f"Розраховано динамічний горизонт для бек-тесту: {lookahead} свічок (макс. довжина кроку: {max_step_duration})")
    return lookahead

def custom_accuracy_scorer(y_true, y_pred):
    """
    Розраховує точність за новою системою бонусів ТІЛЬКИ для успішних угод.
    +2 бали за точне вгадування.
    +1 бал за сусідню зону.
    +0.5 бала за будь-яку іншу успішну зону.
    """
    total_score = 0.0
    
    # Максимально можливий бал - 2 за кожну угоду
    max_possible_score = len(y_true) * 2.0
    if max_possible_score == 0:
        return 0.0

    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == true_label:
            total_score += 2.0
        elif abs(pred_label - true_label) == 1:
            total_score += 1.0
        else:
            total_score += 0.5
    
    # Повертаємо як частку від максимального, щоб було схоже на відсоток (напр. 0.75 -> 75%)
    return total_score / max_possible_score


@numba.jit(nopython=True, cache=True)
def get_sniper_labels_numba(highs, lows, closes, atrs, timesteps, lookahead): # <--- atrs замість xs
    """
    SL = 1 ATR, TP = 3 ATR.
    """
    n = len(closes)
    labels = np.full(n, -1, dtype=np.int8)
    
    for i in range(timesteps, n - 1):
        atr_val = atrs[i]
        if atr_val == 0 or np.isnan(atr_val): continue

        entry_price = closes[i]
        
        # Жорстка математика: 1 до 3 по ATR
        sl_dist = atr_val * 1.0
        tp_dist = atr_val * 3.0
        
        # LONG levels
        long_tp = entry_price + tp_dist
        long_sl = entry_price - sl_dist
        
        # SHORT levels
        short_tp = entry_price - tp_dist
        short_sl = entry_price + sl_dist
        
        is_long_win = False
        is_short_win = False
        
        max_k = min(lookahead, n - 1 - i)
        
        # Check LONG
        for k in range(1, max_k):
            idx = i + k
            if lows[idx] <= long_sl: break
            if highs[idx] >= long_tp:
                is_long_win = True
                break
        
        # Check SHORT
        for k in range(1, max_k):
            idx = i + k
            if highs[idx] >= short_sl: break
            if lows[idx] <= short_tp:
                is_short_win = True
                break
                
        if is_long_win and not is_short_win:
            labels[i] = 1
        elif is_short_win and not is_long_win:
            labels[i] = 0
            
    return labels

def create_hybrid_dataset(df: pd.DataFrame, scaler=None, is_training=False):
    """
    УНІВЕРСАЛЬНА функція підготовки даних для Гібридного Трансформера.
    Гарантує 100% синхронізацію між навчанням, валідацією та ботом.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    # 1. Витягуємо дані
    tokens_data = df[TOKEN_FEATURES].values.astype(np.int32)
    floats_data = df[FLOAT_FEATURES].values.astype(np.float32)
    
    # 2. Скалювання (ТІЛЬКИ для флоатів)
    if is_training:
        # Якщо навчання - фітимо новий скейлер
        scaler = StandardScaler()
        floats_scaled = scaler.fit_transform(floats_data)
    else:
        # Якщо валідація/бот - використовуємо готовий
        if scaler is None: raise ValueError("Scaler required for validation/inference")
        floats_scaled = scaler.transform(floats_data)
        
    # 3. Створення вікон (Sliding Window)
    # Форма: (N, TIMESTEPS, Features)
    tokens_windows = sliding_window_view(tokens_data, window_shape=(TIMESTEPS, len(TOKEN_FEATURES)))
    floats_windows = sliding_window_view(floats_scaled, window_shape=(TIMESTEPS, len(FLOAT_FEATURES)))
    
    # Прибираємо зайву вісь, яку створює sliding_window_view
    tokens_windows = tokens_windows.squeeze(axis=1)
    floats_windows = floats_windows.squeeze(axis=1)
    
    # 4. Підготовка 4-х окремих входів для Keras
    # [Zone, Speed, Vol, Floats]
    X_zone = tokens_windows[:, :, 0]
    X_speed = tokens_windows[:, :, 1]
    X_vol = tokens_windows[:, :, 2]
    X_floats = floats_windows
    
    # Повертаємо X (список входів) та скейлер
    return [X_zone, X_speed, X_vol, X_floats], scaler

def run_training_only(args):
    """
    Режим навчання з фокусом на "Value Entry" (Снайпер).
    Відрізняється від run_training_process тим, що фільтрує входи
    і вчить тільки на чітких сетапах.
    """
    import gc
    main_logger.info(f"--- ЗАПУСК ТРЕНУВАННЯ 'VALUE SNIPER' для {args.symbol} ---")
    
    # 1. Ініціалізація та дані
    temp_bot_instance = TradingBotInstance(symbol=args.symbol, interval="5m", api_key="DUMMY", api_secret="DUMMY", is_backtest=True)
    
    # Беремо буфер для індикаторів
    buffer_start_dt = pd.to_datetime(args.train_start) - pd.Timedelta(days=5) 
    full_df_raw = get_all_historical_futures_data(symbol=args.symbol, interval="5m", start_str=str(buffer_start_dt), end_str=args.train_end)
    
    main_logger.info("Розрахунок індикаторів (New Logic)...")
    full_df = temp_bot_instance._calculate_indicators(full_df_raw)
    
    # Обрізаємо по дату старту
    train_df = full_df[full_df.index >= pd.to_datetime(args.train_start)]
    
    # 2. РОЗМІТКА (Labeling) - "Sniper Logic"
    # Тут ми визначаємо, чи був вхід успішним
    # В майбутньому тут буде виклик складної функції розмітки.
    # Поки що: Проста розмітка для перевірки архітектури
    main_logger.info("Генерація міток (Sniper Logic)...")
    labels = (train_df['close'].shift(-5) > train_df['close']).astype(int).values 
    
    # 3. Підготовка Гібридних Входів
    X_list, scaler = create_hybrid_dataset(train_df, is_training=True)
    
    # Вирівнювання довжини (X коротший через вікно TIMESTEPS)
    labels = labels[TIMESTEPS-1:]
    min_len = min(len(X_list[0]), len(labels))
    X_list = [x[:min_len] for x in X_list]
    y_train = labels[:min_len]
    
    # 4. Збереження скейлера (УНІКАЛЬНЕ ІМ'Я)
    scaler_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{args.symbol}{SCALER_FILE_SUFFIX}")
    joblib.dump(scaler, scaler_path)
    
    # 5. Навчання
    model = build_hybrid_transformer_model(TIMESTEPS, NUM_FLOAT_FEATURES)
    
    # Збереження моделі (УНІКАЛЬНЕ ІМ'Я)
    model_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{args.symbol}{MODEL_FILE_SUFFIX}")
    
    # Ваги класів (для балансування)
    from sklearn.utils import class_weight
    unique_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    model.fit(
        X_list, y_train, 
        epochs=100, 
        batch_size=64, 
        validation_split=0.15,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True)
        ]
    )
    main_logger.info(f"✅ Модель навчена і збережена: {model_path}")

def run_validation_only(args):
    """
    ВАЛІДАЦІЯ (Backtest Simulation) з ДІАГНОСТИКОЮ.
    Показує статистику ймовірностей, щоб зрозуміти, чому 0 угод.
    """
    # 1. Завантаження ключів (для Телеграм)
    load_dotenv(dotenv_path="key.env")
    tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
    tg_chat = os.getenv('YOUR_TELEGRAM_CHAT_ID')

    main_logger.info(f"--- ЗАПУСК ВАЛІДАЦІЇ 'HYBRID SNIPER' для {args.symbol} ---")
    
    scaler_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{args.symbol}{SCALER_FILE_SUFFIX}")
    model_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{args.symbol}{MODEL_FILE_SUFFIX}")
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        sys.exit(f"❌ Не знайдено файли моделі або скейлера для {args.symbol}. Спочатку запустіть навчання.")
        
    scaler = joblib.load(scaler_path)
    custom_objects = {'PositionalEncoding': PositionalEncoding, 'focal_loss_fixed': focal_loss()} 
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # 2. Ініціалізація бота
    temp_bot = TradingBotInstance(
        symbol=args.symbol, interval="5m", api_key="DUMMY", api_secret="DUMMY", 
        is_backtest=True, telegram_bot_token=tg_token, telegram_chat_id_str=tg_chat
    )
    
    # Буфер і дані
    buffer_start = pd.to_datetime(args.test_start) - pd.Timedelta(hours=12)
    df_raw = get_all_historical_futures_data(args.symbol, "5m", start_str=str(buffer_start), end_str=args.test_end)
    
    if df_raw.empty:
        main_logger.error(f"❌ ОТРИМАНО ПОРОЖНІЙ ДАТАФРЕЙМ! Перевірте дати: {args.test_start} - {args.test_end}")
        sys.exit(1)

    main_logger.info("Розрахунок індикаторів...")
    full_df = temp_bot._calculate_indicators(df_raw)
    
    test_df = full_df[full_df.index >= pd.to_datetime(args.test_start)]
    if test_df.empty: sys.exit("❌ Тестовий період пустий після фільтрації дат.")
    
    main_logger.info(f"✅ Дані завантажено. Свічок для тесту: {len(test_df)}")

    # 3. Підготовка та Прогноз
    X_list, _ = create_hybrid_dataset(test_df, scaler=scaler, is_training=False)
    
    main_logger.info(f"Генерація прогнозів для {len(X_list[0])} точок...")
    # !!! ВИПРАВЛЕННЯ: verbose=0, щоб не забивати буфер довгою стрічкою прогрес-бару
    probs = model.predict(X_list, batch_size=1024, verbose=0).flatten()
    
    # --- ДІАГНОСТИКА МОДЕЛІ (ЧОМУ 0 УГОД?) ---
    max_prob = probs.max() if len(probs) > 0 else 0
    avg_prob = probs.mean() if len(probs) > 0 else 0
    min_prob = probs.min() if len(probs) > 0 else 0
    
    print("\n" + "="*50)
    print(f"🔍 ДІАГНОСТИКА МОДЕЛІ ({args.symbol})")
    print(f"📊 Мін. впевненість: {min_prob:.4f}")
    print(f"📊 Середня впевненість: {avg_prob:.4f}")
    print(f"🚀 МАКС. ВПЕВНЕНІСТЬ: {max_prob:.4f}")
    
    CONFIDENCE = 0.45
    if max_prob < CONFIDENCE:
        print(f"⚠️ УВАГА: Модель жодного разу не була впевнена > {CONFIDENCE*100}%.")
        print("   Спробуйте знизити поріг або перенавчити модель.")
    print("="*50 + "\n")

    # 4. Симуляція
    closes = test_df['close'].values
    highs = test_df['high'].values
    lows = test_df['low'].values
    active_steps = test_df['active_step'].values 
    
    trades = 0
    wins = 0
    skip_until = 0
    
    print(f"СИМУЛЯЦІЯ ТОРГІВЛІ (Поріг: {CONFIDENCE})...")

    for i in range(len(probs)):
        real_idx = i + TIMESTEPS - 1
        if real_idx >= len(closes) - 1: break 
        if real_idx < skip_until: continue
        
        prob = probs[i]
        signal = 0
        if prob > CONFIDENCE: signal = 1
        
        if signal != 0:
            entry_price = closes[real_idx]
            current_step = active_steps[real_idx]
            
            # Value Entry: 20% retracement
            retracement = current_step * 0.2
            
            # LONG ONLY (якщо модель вчилась на long)
            tp_price = entry_price - retracement + (current_step * 1.5)
            sl_price = entry_price - retracement - (current_step * 0.65)
            
            outcome = "OPEN"
            for k in range(1, 200):
                fut_idx = real_idx + k
                if fut_idx >= len(closes): break
                c_high = highs[fut_idx]
                c_low = lows[fut_idx]
                
                if c_low <= sl_price:
                    outcome = "LOSS"
                    break
                if c_high >= tp_price:
                    outcome = "WIN"
                    break
            
            if outcome != "OPEN":
                trades += 1
                if outcome == "WIN": wins += 1
                skip_until = fut_idx 

    win_rate = (wins / trades * 100) if trades > 0 else 0
    
    # !!! ВИПРАВЛЕННЯ: Додано спеціальний заголовок, який шукає менеджер
    print("\n" + "ФІНАЛЬНЕ РЕЗЮМЕ (ВАЛІДАЦІЯ)") 
    print("=========================================================================================================================")
    print(f"РЕЗУЛЬТАТ ВАЛІДАЦІЇ: {args.symbol}")
    print(f"Всього угод: {trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print("=========================================================================================================================")

    # 5. Збереження
    performance_file = os.path.join(SCRIPT_DIR_NEW_LOGIC, "model_performance.json")
    data = {}
    if os.path.exists(performance_file):
        try:
            with open(performance_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception: data = {}

    if args.symbol not in data: data[args.symbol] = {}
    
    # !!! ВИПРАВЛЕННЯ: Зберігаємо і для Ensemble, і для Transformer, щоб звіт був повний
    stats_block = {
        'correct': wins,
        'total': trades,
        'win_rate': win_rate,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    data[args.symbol]['Ensemble'] = stats_block
    data[args.symbol]['Transformer'] = stats_block # <--- Дублюємо для коректного відображення

    try:
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        main_logger.info(f"✅ Звіт збережено: {performance_file}")
        
        # Відправка в ТГ
        report_msg = temp_bot._format_performance_report(data, args.symbol)
        # Додаємо діагностику в повідомлення
        report_msg += f"\n🔍 <i>Max Prob: {max_prob:.4f} (Threshold: {CONFIDENCE})</i>"
        temp_bot._send_telegram_report(report_msg)
        
    except Exception as e:
        main_logger.error(f"❌ Помилка збереження/відправки: {e}")

def run_training_process(args):
    """
    ПОВНИЙ ЦИКЛ 1:1 (Гібридний Трансформер).
    """
    main_logger.info(f"--- ЗАПУСК ПОВНОГО ЦИКЛУ (Hybrid Transformer) для {args.symbol} ---")

    # 1. Завантаження (Тільки 5m)
    main_df_raw = get_all_historical_futures_data(symbol=args.symbol, interval="5m", lookback_months=args.lookback_months)
    if main_df_raw.empty: sys.exit("Не вдалося завантажити дані.")

    # 2. Індикатори (Єдине джерело правди)
    main_logger.info("Розрахунок індикаторів (New Logic)...")
    temp_bot = TradingBotInstance(symbol=args.symbol, interval="5m", api_key="DUMMY", api_secret="DUMMY", is_backtest=True)
    full_df_calc = temp_bot._calculate_indicators(main_df_raw)
    
    # Видаляємо NaN на початку (перші 20-50 свічок, де SMA/ATR ще рахувались)
    full_df_calc.dropna(inplace=True)

    # 3. Спліт 80/20 (Хронологічний)
    split_index = int(len(full_df_calc) * 0.80)
    train_df = full_df_calc.iloc[:split_index]
    test_df = full_df_calc.iloc[split_index:]
    
    main_logger.info(f"Train size: {len(train_df)}, Val size: {len(test_df)}")

    # 4. Підготовка X (Inputs)
    main_logger.info("Підготовка тензорів для навчання...")
    
    # X_train_list = [Zone, Speed, Vol, Floats]
    X_train_list, scaler = create_hybrid_dataset(train_df, is_training=True)
    
    # Зберігаємо скейлер з УНІКАЛЬНИМ ім'ям для символу
    scaler_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{args.symbol}{SCALER_FILE_SUFFIX}")
    joblib.dump(scaler, scaler_path)
    main_logger.info(f"✅ Скейлер збережено: {scaler_path}")
    
    # 5. Підготовка y (Labels) - "Direction" Logic (як база для перевірки архітектури)
    # Нам треба змістити лейбли, бо X[i] закінчується на i, а y[i] має бути майбутнім
    y_raw = (train_df['close'].shift(-1) > train_df['close']).astype(int).values
    # Обрізаємо y під розмір X (перші TIMESTEPS-1 відрізаються вікном)
    y_train = y_raw[TIMESTEPS-1:]
    
    # Перевірка розмірностей
    if len(X_train_list[0]) != len(y_train):
        min_len = min(len(X_train_list[0]), len(y_train))
        X_train_list = [x[:min_len] for x in X_train_list]
        y_train = y_train[:min_len]

    # Валідаційний сет
    X_val_list, _ = create_hybrid_dataset(test_df, scaler=scaler, is_training=False)
    y_val_raw = (test_df['close'].shift(-1) > test_df['close']).astype(int).values
    y_val = y_val_raw[TIMESTEPS-1:]
    
    min_len_val = min(len(X_val_list[0]), len(y_val))
    X_val_list = [x[:min_len_val] for x in X_val_list]
    y_val = y_val[:min_len_val]

    # 6. Побудова та Навчання Моделі
    main_logger.info("Компіляція Гібридного Трансформера...")
    # Імпортуємо build_hybrid_transformer_model (вона вже має бути в файлі)
    model = build_hybrid_transformer_model(TIMESTEPS, NUM_FLOAT_FEATURES)
    
    # Зберігаємо модель з УНІКАЛЬНИМ ім'ям
    model_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{args.symbol}{MODEL_FILE_SUFFIX}")
    
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    
    main_logger.info("Початок навчання...")
    model.fit(
        x=X_train_list, # Список з 4 масивів
        y=y_train, 
        epochs=50, 
        batch_size=64, 
        validation_data=(X_val_list, y_val), 
        callbacks=[es, mc], 
        verbose=1
    )
    main_logger.info(f"✅ Модель збережено: {model_path}")
    
    # 7. Авто-Валідація
    val_args = argparse.Namespace(
        symbol=args.symbol,
        test_start=test_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        test_end=test_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        no_progress_bar=args.no_progress_bar
    )
    run_validation_only(val_args)
    
# --- Numba JIT compiled function ---
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

@numba.jit(nopython=True, nogil=True)
def calculate_step_index(step_level_arr: np.ndarray) -> np.ndarray:
    """
    Виправлена версія, що базується на логіці з calculate_step_count_from_user_logic.
    Нарощує лічильник на початку нового тренду та при його продовженні.
    Скидає лічильник на 1 при розвороті тренду.
    Зберігає значення лічильника, якщо рівень не змінюється.
    """
    n = len(step_level_arr)
    if n == 0:
        return np.zeros(n, dtype=np.int64)

    step_indices = np.zeros(n, dtype=np.int64)
    trend_dir = 0    # 0 = Невизначено, 1 = Вгору, -1 = Вниз
    step_index = 0

    for i in range(n):
        if i == 0:
            step_index = 1
            step_indices[i] = step_index
            # Напрямок на першому кроці ще не визначено
            continue

        # Визначаємо напрямок поточного руху
        current_dir = 0
        if step_level_arr[i] > step_level_arr[i-1]:
            current_dir = 1
        elif step_level_arr[i] < step_level_arr[i-1]:
            current_dir = -1

        # Якщо рух відсутній (флет), лічильник і напрямок тренду не змінюються
        if current_dir == 0:
            step_indices[i] = step_index
            continue
        
        # Якщо тренд тільки починається (був невизначений) або продовжується
        if trend_dir == 0 or current_dir == trend_dir:
            step_index += 1
        # Інакше - відбувся розворот тренду
        else:
            step_index = 1
        
        # Оновлюємо стан для наступної ітерації
        trend_dir = current_dir
        step_indices[i] = step_index
        
    return step_indices
@numba.jit(nopython=True, nogil=True)
def _calculate_pivot_step_index_price_based(step_level_arr: np.ndarray, x_arr: np.ndarray, reset_threshold: int = 3) -> np.ndarray:
    """
    ЧЕРВОНИЙ ІНДЕКС (V8 - PIVOT SWITCH).
    Реалізація логіки з вашого прикладу:
    1. Якщо відкат від High >= 3 кроків -> Тренд стає DOWN, Pivot стає High.
    2. Тепер індекс = (PivotHigh - Price) / X.
    3. Тому при русі 1090 (High) -> 1060 індекс = 3.
    4. А при відскоку 1060 -> 1070 індекс = 2 (бо ближче до High).
    """
    n = len(step_level_arr)
    pivot_step_index_out = np.zeros(n, dtype=np.int64)
    
    if n == 0: return pivot_step_index_out

    # 1 = UP Trend (Міряємо від Low), -1 = DOWN Trend (Міряємо від High)
    # За замовчуванням починаємо, наче ми в UP, і наш півот - це старт
    wave_direction = 1 
    pivot_price = step_level_arr[0] # Це наша "Точка нуль"
    
    # Змінна для трекінгу екстремуму всередині поточного руху (щоб зловити розворот)
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

        # Оновлюємо екстремум поточного руху
        if wave_direction == 1: # Якщо ми йдемо ВГОРУ
            if current_price > current_extreme:
                current_extreme = current_price
        else: # Якщо ми йдемо ВНИЗ
            if current_price < current_extreme:
                current_extreme = current_price

        # Перевірка на розворот (Trigger)
        # Якщо ми в UP, але ціна впала від Максимуму на 3 кроки
        if wave_direction == 1:
            dist_from_high = (current_extreme - current_price) / current_x
            if dist_from_high >= reset_threshold:
                # ПЕРЕМИКАННЯ! Тепер ми падаємо.
                wave_direction = -1
                pivot_price = current_extreme # Наш новий нуль - це Хай
                current_extreme = current_price # Скидаємо локальний екстремум

        # Якщо ми в DOWN, але ціна виросла від Мінімуму на 3 кроки
        elif wave_direction == -1:
            dist_from_low = (current_price - current_extreme) / current_x
            if dist_from_low >= reset_threshold:
                # ПЕРЕМИКАННЯ! Тепер ми ростемо.
                wave_direction = 1
                pivot_price = current_extreme # Наш новий нуль - це Лоу
                current_extreme = current_price

        # ФІНАЛЬНИЙ РОЗРАХУНОК
        # Індекс - це завжди відстань від Актуального Півоту (High або Low)
        # round() використовується, щоб 2.99 стало 3, як у вашому прикладі
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
                print(f"CONSOLE ERROR (TelegramLoggingHandler): Invalid TELEGRAM_CHAT_ID_FOR_LOGGING: {chat_id_str}. Telegram logging will be disabled for this handler.")
        if not self.token:
            print("CONSOLE ERROR (TelegramLoggingHandler): Token for Telegram logging not provided. Telegram logging will be disabled for this handler.")

    def emit(self, record):
        if not self.token or not self.chat_id:
            return
            
        log_entry = self.format(record)
        
        # --- ПОЧАТОК НОВОГО ФІЛЬТРУ ---

        # 1. Повністю ігноруємо детальні, але зашумлені повідомлення про оновлення ордерів
        if "ORDER_TRADE_UPDATE for" in log_entry:
            return

        # 2. Визначаємо ключові слова, які ТОЧНО мають бути надіслані
        # Це помилки, попередження та найважливіші події.
        essential_keywords = [
            "!!!",                  # Виконання SL/TP
            "filled",               # Підтвердження входу в позицію
            "action: new entry",    # Підтвердження виставлення нового лімітного ордера
            "critical",             # Критичні помилки
            "error",                # Помилки
            "failed",               # Помилки
            "unprotected",          # Позиція без захисту
            "stopped due to",       # Зупинка бота
            "завершив роботу",      # Завершення роботи
            "сигнал відхилено",     # Сигнал відхилено
            "ML_FILTER"             # <--- АБО ЦЕЙ РЯДОК
        ]

        # Перевіряємо, чи має повідомлення бути надісланим
        should_send = record.levelno >= logging.WARNING or \
                      any(keyword.lower() in log_entry.lower() for keyword in essential_keywords)

        if not should_send:
            return # Якщо повідомлення не важливе - не надсилаємо його
            
        # --- КІНЕЦЬ НОВОГО ФІЛЬТРУ ---

        log_entry_with_prefix = f"[{self.symbol_prefix}] {log_entry}"

        if len(log_entry_with_prefix) > 4096:
            log_entry_with_prefix = log_entry_with_prefix[:4090] + "\n... (truncated)"
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': f"[{self.symbol_prefix}] {log_entry}", # Просто об'єднуємо рядок без Markdown
            # 'parse_mode': None # ВСТАНОВЛЮЄМО В NONE АБО ВЗАГАЛІ ВИДАЛЯЄМО
        }
        try:
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            # self.logger.debug(f"Telegram Handler: Message sent successfully. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            # Логуйте тут повний текст відповіді, якщо доступно
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

def get_fixed_asset_grid(df: pd.DataFrame, full_grid: np.ndarray, num_neighbors: int = 4) -> np.ndarray:
    """
    Повертає ФІКСОВАНУ кількість кроків сітки для активу.
    Центрується по медіанному ATR.
    Розмір виходу завжди = 1 (Center) + num_neighbors*2.
    При num_neighbors=4 це буде 9 сіток.
    """
    if df.empty or 'atr' not in df.columns:
        # Повертаємо середню частину сітки як заглушку
        mid = len(full_grid) // 2
        start = max(0, mid - num_neighbors)
        return full_grid[start : start + 1 + num_neighbors*2]

    # 1. Знаходимо "Центральний" крок для цього активу (медіана)
    median_atr = df['atr'].median()
    if np.isnan(median_atr) or median_atr == 0:
        target_step = full_grid[len(full_grid)//2]
    else:
        target_step = median_atr * 3.0

    # 2. Знаходимо індекс цього кроку в повній сітці
    # np.abs шукає найближчий елемент
    center_idx = (np.abs(full_grid - target_step)).argmin()

    # 3. Формуємо вікно (Центр +/- neighbors)
    # Гарантуємо, що індекси не вийдуть за межі
    max_idx = len(full_grid) - 1
    
    start_idx = center_idx - num_neighbors
    end_idx = center_idx + num_neighbors
    
    # Якщо вийшли за ліву межу (індекс < 0) - зсуваємо вікно вправо
    if start_idx < 0:
        shift = abs(start_idx)
        start_idx += shift
        end_idx += shift
        
    # Якщо вийшли за праву межу - зсуваємо вліво
    if end_idx > max_idx:
        shift = end_idx - max_idx
        start_idx -= shift
        end_idx -= shift
        
    # Фінальний захист меж (якщо сітка дуже мала)
    start_idx = max(0, start_idx)
    end_idx = min(max_idx, end_idx)

    # Вирізаємо
    return full_grid[start_idx : end_idx + 1]

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

        # --- ДОДАЙТЕ ЦЕЙ РЯДОК ТУТ ---
        self.step_coeff = 3.0  # Значення за замовчуванням
        # -----------------------------


        # --- ЗМІНА 1: Ініціалізуємо нові атрибути для системи звітності ---
        self.pending_prediction_for_report = None
        self.performance_file_path = os.path.join(SCRIPT_DIR, "model_performance.json")
        self.performance_file_lock = threading.Lock()
        # --- КІНЕЦЬ ЗМІНИ ---

        # --- НАЛАШТУВАННЯ ЛОГГЕРА ---
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

        self.logger.info(f"Ініціалізація TradingBotInstance для {self.symbol} на інтервалі {self.interval}")


        # --- ЗАВАНТАЖЕННЯ МОДЕЛІ (ТІЛЬКИ TRANSFORMER) ---
        self.model = None
        self.scaler = None
        
        if not is_backtest:
            try:
                scaler_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{self.symbol}_StepPredictor_Scaler_3D.joblib")
                model_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{self.symbol}_StepPredictor_Transformer_model.keras")
                
                self.scaler = joblib.load(scaler_path)
                custom_objects = {'PositionalEncoding': PositionalEncoding, 'focal_loss_fixed': focal_loss()}
                self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                self.logger.info("✅ Transformer та 3D-Scaler успішно завантажено.")
            except Exception as e:
                self.logger.critical(f"❌ Критична помилка завантаження моделі: {e}")
                self.should_stop_bot_flag_instance = True

        # --- ПОВНИЙ НАБІР ПАРАМЕТРІВ ---
        self.base_tp_multiplier = Decimal(str(self.config.get('TP_MULTIPLIER', '1.5')))
        self.base_sl_multiplier = Decimal(str(self.config.get('SL_MULTIPLIER', '1.0')))
        self.base_entry_retracement_pct = Decimal(str(self.config.get('ENTRY_RETRACEMENT_PCT', '0.1')))
        self.zone_1_params = self.config.get('ZONE_1_PARAMS', {})
        self.zone_2_params = self.config.get('ZONE_2_PARAMS', {})
        self.zone_3_params = self.config.get('ZONE_3_PARAMS', {})
        self.logger.info(f"Завантажено специфічні параметри для зон: Зона1? {'Так' if self.zone_1_params else 'Ні'}, Зона2? {'Так' if self.zone_2_params else 'Ні'}, Зона3? {'Так' if self.zone_3_params else 'Ні'}")
        
        self.probability_param = float(self.config.get('PROBABILITY_PARAM', 150.0))
        self.target_percent_param = float(self.config.get('TARGET_PERCENT_PARAM', 20.0))
        # self.selected_timeframe_higher = self.config.get('SELECTED_TIMEFRAME_HIGHER', '2h')
        self.limit_main_tf = int(self.config.get('LIMIT_MAIN_TF', 1500))
        # self.limit_higher_tf = int(self.config.get('LIMIT_HIGHER_TF', 1500))
        self.atr_period = int(self.config.get('ATR_PERIOD', 140))
        self.residual_smooth = int(self.config.get('RESIDUAL_SMOOTH', 10))
        self.calculate_vol_zones = self.config.get('CALCULATE_VOL_ZONES', True)
        self.dc_period = int(self.config.get('DC_PERIOD', 50))
        self.dc_quantile_lookback = int(self.config.get('DC_QUANTILE_LOOKBACK', 1500))
        self.dc_low_quantile = float(self.config.get('DC_LOW_quantile', 0.25))
        self.dc_high_quantile = float(self.config.get('DC_HIGH_quantile', 0.80))
        self.min_tp_multiplier = Decimal(str(self.config.get('MIN_TP_MULTIPLIER', '0.05')))
        self.max_tp_multiplier = Decimal(str(self.config.get('MAX_TP_MULTIPLIER', '4.0')))
        self.min_sl_multiplier = Decimal(str(self.config.get('MIN_SL_MULTIPLIER', '0.05')))
        self.max_sl_multiplier = Decimal(str(self.config.get('MAX_SL_MULTIPLIER', '0.9')))
        self.initial_capital_per_trade_usdt = Decimal(str(self.config.get('INITIAL_CAPITAL_PER_TRADE_USDT', '1.0')))
        self.virtual_bot_capital_usdt = Decimal(str(self.config.get('VIRTUAL_BOT_CAPITAL_USDT', '1.0')))
        self.bot_starting_virtual_capital = self.virtual_bot_capital_usdt
        self.bot_virtual_capital_stop_loss_percent = Decimal(str(self.config.get('BOT_VIRTUAL_CAPITAL_STOP_LOSS_PERCENT', '0.60')))
        self.total_account_stop_loss_usdt = Decimal(str(self.config.get('TOTAL_ACCOUNT_STOP_LOSS_USDT', '0.0')))
        self.min_distance_from_entry_ticks = Decimal(str(self.config.get('MIN_DISTANCE_FROM_ENTRY_TICKS', '5')))
        self.liquidation_price_buffer_ticks = Decimal(str(self.config.get('LIQUIDATION_PRICE_BUFFER_TICKS', '10')))
        self.max_risk_percent_of_per_trade_capital = Decimal(str(self.config.get('MAX_RISK_PERCENT_OF_PER_TRADE_CAPITAL', '0.7')))
        self.extra_confidence_threshold_for_logging = Decimal(str(self.config.get('EXTRA_CONFIDENCE_LOGGING', '0.85')))
        self.prediction_stability_period = int(self.config.get('PREDICTION_STABILITY_PERIOD', 3))
        self.min_prediction_diff = Decimal(str(self.config.get('MIN_PREDICTION_DIFFERENCE', '0.10'))) # Мінімальна різниця 10%

        # Створюємо буфер для зберігання останніх прогнозів
        self.prediction_history = deque(maxlen=self.prediction_stability_period)
        self.cancel_on_opposite_signal_threshold = Decimal(str(self.config.get('CANCEL_ON_OPPOSITE_SIGNAL_THRESHOLD', '0.78')))

        self.client = None 
        self.ws_client = None 
        self.listen_key = None
        self.klines_df_main = pd.DataFrame()
        # self.klines_df_higher_tf = pd.DataFrame()
        self.last_processed_kline_open_time = None
        self.symbol_trading_rules = {}
        self.leverage_brackets_for_symbol = []
        self._reset_active_limit_entry_order_details()
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
        self.last_signal_timestamp = None
        self.last_listen_key_renewal_time = 0
        self.stop_flag_file_path_instance = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"STOP_TRADING_BOT_{self.symbol}.flag")
        self.last_heartbeat_log_time = 0
        self.start_time = time.time()
        self.is_processing_position_close = False # <-- ДОДАЙ ЦЕЙ РЯДОК
        self.has_dumped_realtime_data = False # <--- ★★★ ДОДАЙ ЦЕЙ РЯДОК ★★★
        
        self.logger.info("---------------------------------------------------------------------")
        self.logger.info(f"НОВА СЕСІЯ БОТА для {self.symbol} на {self.interval}")
        self.logger.info(f"Налаштоване значення TOTAL_ACCOUNT_STOP_LOSS_USDT (глобальна дозволена просадка): {float(self.total_account_stop_loss_usdt):.4f}")
        self.logger.info("---------------------------------------------------------------------")

        if is_backtest:
            # --- Спеціальна ініціалізація для бек-тесту ---
            self.logger.info("Ініціалізація бота в режимі БЕК-ТЕСТУВАННЯ.")
            try:
                # Створюємо клієнт тільки для публічних даних (правила торгівлі)
                self.client = UMFutures()
                if not self._get_symbol_trading_rules():
                    raise ConnectionError("Не вдалося завантажити торгові правила для бек-тесту.")
                self.logger.info("Публічні торгові правила для бек-тесту успішно завантажено.")
                # Ми більше не намагаємося завантажити правила плеча тут
            except Exception as e:
                self.logger.error(f"Критична помилка ініціалізації для бек-тесту: {e}")
                raise
        else:
            # Для реальної торгівлі викликаємо повний метод ініціалізації
            self._initialize_binance_client()

                
    def _reset_active_limit_entry_order_details(self):
        self.active_limit_entry_order_details = {
            "orderId": None, "clientOrderId": None, "symbol": None, "order_side": None,
            "position_side": None, "price": 0.0, "orig_qty": 0.0, "executed_qty": 0.0,
            "status": None, "prev_x_at_signal": 0.0, "qty_precision": 0,
            "price_precision_asset": 0, "tick_size_str": "", "signal_candle_timestamp": None,
            "step_level_at_signal": None
        }

    def _initialize_binance_client(self):
        if not self.api_key or not self.api_secret:
            self.logger.error("API key або secret не надано для ініціалізації Binance клієнта.")
            raise ValueError("API ключі відсутні для ініціалізації Binance клієнта.")
        try:
            self.client = UMFutures(key=self.api_key, secret=self.api_secret)
            self.logger.info("Binance UMFutures клієнт (РЕАЛЬНИЙ РИНОК) успішно ініціалізовано.")
            self.client.ping()
            self.logger.info("З'єднання з Binance API успішне (ping).")
            server_time = self.client.time()
            self.logger.info(f"Час сервера Binance: {pd.to_datetime(server_time['serverTime'], unit='ms')}")
        except ClientError as e:
            self.logger.error(f"Помилка Binance API Client під час ініціалізації: Status {e.status_code}, Code {e.error_code}, Message: {e.error_message}")
            raise ConnectionError(f"Не вдалося ініціалізувати Binance клієнт через помилку API: {e.error_message}")
        except Exception as e:
            self.logger.error(f"Загальна помилка ініціалізації Binance клієнта: {e}", exc_info=True)
            raise ConnectionError(f"Не вдалося ініціалізувати Binance клієнт через загальну помилку: {e}")

    def _send_final_telegram_message(self, message: str):
        if self.telegram_bot_token_for_logging and self.telegram_chat_id_for_logging_str:
            chat_id_int = None
            try:
                chat_id_int = int(self.telegram_chat_id_for_logging_str)
            except ValueError:
                self.logger.error("Некоректний Chat ID для фінального повідомлення Telegram.")
                return

            final_message_with_prefix = f"[{self.symbol}] {message}"
            url = f"https://api.telegram.org/bot{self.telegram_bot_token_for_logging}/sendMessage"
            payload = { 'chat_id': chat_id_int, 'text': final_message_with_prefix, 'parse_mode': 'Markdown'}
            try:
                response = requests.post(url, data=payload, timeout=10)
                response.raise_for_status()
                self.logger.info("Фінальне повідомлення про статус бота надіслано в Telegram.")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Помилка надсилання фінального повідомлення в Telegram: {e}")
            except Exception as e_unknown:
                self.logger.error(f"Невідома помилка надсилання фінального повідомлення в Telegram: {e_unknown}", exc_info=True)
        else:
            self.logger.warning("Токен Telegram або Chat ID не налаштовано для фінального повідомлення.")


    def _get_symbol_trading_rules(self, retries: int = 3, delay: int = 5):
        # ... (код методу як у вашій версії) ...
        for attempt in range(retries):
            try:
                self.logger.info(f"Запит exchangeInfo (Спроба {attempt + 1}/{retries}) для {self.symbol}...")
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
                        
                        self.logger.info(f"Отримано торгові правила для {self.symbol}: "
                                    f"qtyPrec={rules['quantityPrecision']}, pricePrec={rules['pricePrecision']}, "
                                    f"minQty={rules['minQty']}, stepSize={rules['stepSize']}, "
                                    f"minNotional={rules['minNotional']}, tickSize={rules['tickSize']}")
                        self.symbol_trading_rules = rules 
                        return True 
                self.logger.warning(f"Символ {self.symbol} не знайдено в exchangeInfo (Спроба {attempt + 1}/{retries}).")
                # Не повертаємо False одразу, спробуємо ще
                if attempt < retries - 1: time.sleep(delay)
                else: return False
            except ClientError as e:
                self.logger.error(f"API ClientError в _get_symbol_trading_rules (Спроба {attempt + 1}/{retries}) для {self.symbol}: "
                             f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.headers}")
                if attempt < retries - 1: time.sleep(delay)
                else: return False
            except requests.exceptions.RequestException as e: 
                self.logger.error(f"Мережева помилка в _get_symbol_trading_rules (Спроба {attempt + 1}/{retries}) для {self.symbol}: {e}")
                if attempt < retries - 1: time.sleep(delay * (attempt + 1))
                else: return False
            except Exception as e:
                self.logger.error(f"Неочікувана помилка в _get_symbol_trading_rules (Спроба {attempt + 1}/{retries}) для {self.symbol}: {e}", exc_info=True)
                return False # Не повторювати при невідомій помилці
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
            self.logger.info(f"Attempting to set margin type for {self.symbol} to ISOLATED.")
            self.client.change_margin_type(symbol=self.symbol, marginType='ISOLATED')
            self.logger.info(f"Margin type for {self.symbol} successfully set to ISOLATED.")
            return True
        except ClientError as e:
            if e.error_code == -4046: # "No need to change margin type."
                self.logger.info(f"Margin type for {self.symbol} is already ISOLATED or cannot be changed now. Considered successful.")
                return True
            self.logger.error(f"API ClientError setting ISOLATED margin for {self.symbol}: "
                         f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.headers}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error setting ISOLATED margin for {self.symbol}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error setting ISOLATED margin for {self.symbol}: {e}", exc_info=True)
            return False

    
    def _get_account_balance_usdt(self, asset: str = 'USDT', retries: int = 3, delay: int = 5):
        for i in range(retries):
            try:
                balances = self.client.balance(recvWindow=6000)
                for acc_balance in balances:
                    if acc_balance['asset'] == asset:
                        return float(acc_balance['balance'])
                self.logger.warning(f"Актив {asset} не знайдено в балансі ф'ючерсного акаунту ({self.symbol}, спроба {i+1}/{retries}).")
                if i < retries - 1: time.sleep(delay)
                else: return None 
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e_net:
                log_level = logging.INFO if i < retries - 1 else logging.WARNING # Змінено на INFO/WARNING
                self.logger.log(log_level, f"Мережева помилка отримання балансу ({self.symbol}, спроба {i+1}/{retries}): {type(e_net).__name__} - {e_net}")
                if i < retries - 1: self.logger.debug(f"Повторна спроба через {delay} секунд...") # DEBUG для меншого шуму
                else: self.logger.error(f"Всі {retries} спроби отримати баланс для {self.symbol} провалені (мережева помилка).")
                if i < retries - 1: time.sleep(delay)
                else: return None
            except ClientError as e_client:
                log_level = logging.WARNING if i < retries - 1 else logging.ERROR
                self.logger.log(log_level, f"Помилка API Binance при отриманні балансу ({self.symbol}, спроба {i+1}/{retries}): Status {e_client.status_code}, Code {e_client.error_code}, Msg: {e_client.error_message}")
                if i < retries - 1: self.logger.info(f"Повторна спроба через {delay} секунд...")
                else: self.logger.error(f"Всі {retries} спроби отримати баланс для {self.symbol} провалені (помилка API).")
                if i < retries - 1: time.sleep(delay)
                else: return None
            except Exception as e_other:
                self.logger.error(f"Непередбачена помилка отримання балансу ({self.symbol}, спроба {i+1}/{retries}): {e_other}", exc_info=True)
                if i < retries - 1: time.sleep(delay)
                else: return None
        return None

    def _check_capital_and_stop_if_needed(self, reason: str):
        """Перевіряє, чи не досяг віртуальний капітал порогу зупинки."""
        # Розраховуємо поріг зупинки: початковий капітал * (1 - відсоток просадки)
        stop_threshold = self.bot_starting_virtual_capital * (Decimal('1.0') - self.bot_virtual_capital_stop_loss_percent)

        if self.virtual_bot_capital_usdt <= stop_threshold:
            self.bot_stopped_due_to_total_sl = True # Встановлюємо головний прапорець зупинки
            self.logger.critical(
                f"!!! STOP LOSS БОТА !!! Капітал {self.virtual_bot_capital_usdt:.4f} USDT досяг або впав нижче порогу {stop_threshold:.4f} USDT."
            )
            self.logger.critical(f"Бот для {self.symbol} зупинено. Причина: {reason}")
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
                self.logger.error(f"API ClientError getting position risk (Attempt {attempt + 1}/{retries}) for {log_msg_symbol}: "
                             f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.headers}")
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
                return None 
        self.logger.error(f"All {retries} attempts failed for getting position risk for {log_msg_symbol}.")
        return None

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
            # --- ★★★ ОСЬ ЦЕ ВИПРАВЛЕННЯ (СИНХРОНІЗАЦІЯ З get_all_historical_futures_data) ★★★ ---
            df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # --- ★★★ КІНЕЦЬ БЛОКУ ★★★ ---

            # Цей рядок (старий 1741) більше не потрібен, видаліть його:
            # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
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
        """Скидає внутрішній стан поточної позиції в боті."""
        self.logger.info(f"Скидання внутрішнього стану позиції для {self.symbol}.")
        self.current_position_on_exchange = None
        self.current_position_entry_price = 0.0
        self.current_position_quantity = 0.0
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.is_processing_position_close = False # <-- І СЮДИ
        
        # Скидаємо також деталі лімітного ордера на вхід, якщо він ще був активний
        if self.active_limit_entry_order_details and self.active_limit_entry_order_details.get("orderId"):
            self._reset_active_limit_entry_order_details()

    def _get_current_position_info(self):
        """Отримує свіжу інформацію про позицію для поточного символу з біржі."""
        try:
            self.logger.debug(f"Запит свіжої інформації про позицію для {self.symbol}...")
            # Використовуємо get_position_risk, оскільки він надає деталі по конкретному символу
            positions = self.client.get_position_risk(symbol=self.symbol)
            if positions:
                # Повертаємо перший (і єдиний) словник зі списку
                return positions[0] 
        except Exception as e:
            self.logger.error(f"Помилка отримання поточної інформації про позицію для {self.symbol}: {e}")
        return None    
    
    def _on_websocket_message(self, ws_client_instance, message):
        if self.bot_stopped_due_to_total_sl or self.should_stop_bot_flag_instance:
            return

        try:
            msg_data = json.loads(message)
            event_type = msg_data.get('e')

            if event_type == 'kline':
                k_data = msg_data['k']
                symbol_ws = k_data['s']
                interval_ws = k_data['i']
                is_candle_closed = k_data['x']
                kline_open_time_ws = pd.to_datetime(k_data['t'], unit='ms')

                if symbol_ws == self.symbol:
                    if interval_ws == self.interval and is_candle_closed:
                        if self.last_processed_kline_open_time is None or kline_open_time_ws > self.last_processed_kline_open_time:
                            self.logger.info(f"Processing closed {self.interval} candle for {self.symbol} at {kline_open_time_ws.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Оновлюємо дані тільки для ОСНОВНОГО ТФ
                            self.klines_df_main = self._update_klines_with_new_data(self.klines_df_main, msg_data, self.limit_main_tf)
                            self.last_processed_kline_open_time = kline_open_time_ws

                            if not self.klines_df_main.empty:
                                # --- ЗМІНА: Збереження (Single TF) ---
                                if not self.has_dumped_realtime_data:
                                    try:
                                        dump_data = {'df_main_raw': self.klines_df_main.copy()}
                                        dump_filename = f"REALTIME_DUMP_{self.symbol}.pkl"
                                        with open(dump_filename, 'wb') as f:
                                            joblib.dump(dump_data, f)
                                        self.logger.info(f"!!! ЗБЕРЕЖЕНО ДАНІ (РЕАЛЬНИЙ БОТ): {dump_filename} !!!")
                                        self.has_dumped_realtime_data = True
                                    except Exception as e_dump:
                                        self.logger.error(f"!!! ПОМИЛКА ЗБЕРЕЖЕННЯ (РЕАЛЬНИЙ БОТ): {e_dump} !!!")
                                # --- КІНЕЦЬ БЛОКУ ЗБЕРЕЖЕННЯ ---

                                # --- ЗМІНА: Передаємо тільки main df ---
                                df_with_indicators = self._calculate_indicators(self.klines_df_main.copy())
                                
                                if not df_with_indicators.empty:
                                    last_kline_processed = df_with_indicators.iloc[-1]
                                    price_prec = self.symbol_trading_rules.get('pricePrecision', 2)
                                    self.logger.info(
                                        f"  -> Candle for Logic {self.symbol}: Close: {last_kline_processed.get('close', float('nan')):.{price_prec}f}, "
                                        f"StepLvl: {last_kline_processed.get('step_level', float('nan')):.{price_prec}f}, "
                                        f"X: {last_kline_processed.get('x', float('nan')):.{price_prec}f}"
                                    )
                                    self._check_and_execute_trade_logic(df_with_indicators)
                    
                    # Гілка для higher_tf ВИДАЛЕНА повністю

            elif event_type == 'ORDER_TRADE_UPDATE':
                order_data = msg_data.get('o', {})
                if order_data.get('s') != self.symbol: return

                order_id_ws = str(order_data.get('i'))
                client_order_id_ws = order_data.get('c')
                order_status_ws = order_data.get('X')
                order_type_ws = order_data.get('ot')
                realized_pnl_str = order_data.get('rp', '0')
                
                self.logger.info(f"ORDER_TRADE_UPDATE for {self.symbol}: OrderID {order_id_ws}, ClientOrderID {client_order_id_ws}, Status {order_status_ws}, Type {order_type_ws}, RPnL: {realized_pnl_str}")

                if self.active_limit_entry_order_details.get("clientOrderId") == client_order_id_ws:
                    self.logger.info(f"Processing update for ACTIVE LIMIT ENTRY order {client_order_id_ws}.")
                    self.active_limit_entry_order_details["status"] = order_status_ws

                    if order_status_ws in ['PARTIALLY_FILLED', 'FILLED']:
                        filled_qty_this_update = float(order_data.get('z', '0.0')) 
                        qty_filled_in_this_event = filled_qty_this_update - self.active_limit_entry_order_details.get("executed_qty", 0.0)

                        if qty_filled_in_this_event > 0:
                            last_trade_price_float = float(order_data.get('L', '0.0')) or float(order_data.get('ap', '0.0'))
                            if last_trade_price_float > 0:
                                self.logger.info(f"Limit entry order {order_id_ws} filled for {qty_filled_in_this_event} at {last_trade_price_float}")
                                last_trade_price_decimal = Decimal(str(last_trade_price_float))

                                if self.current_position_quantity > 0: 
                                    new_total_value = (self.current_position_entry_price * Decimal(str(self.current_position_quantity))) + (last_trade_price_decimal * Decimal(str(qty_filled_in_this_event)))
                                    self.current_position_quantity += qty_filled_in_this_event
                                    self.current_position_entry_price = new_total_value / Decimal(str(self.current_position_quantity))
                                else: 
                                    self.current_position_entry_price = last_trade_price_decimal 
                                    self.current_position_quantity = qty_filled_in_this_event
                                
                                self.current_position_on_exchange = self.active_limit_entry_order_details["position_side"]
                                self.active_limit_entry_order_details["executed_qty"] = filled_qty_this_update
                                self.logger.info(f"POSITION UPDATED: Side: {self.current_position_on_exchange}, Total Qty: {self.current_position_quantity}, Avg Entry Price: {self.current_position_entry_price}")
                                
                                self.current_position_entry_client_order_id = client_order_id_ws

                                order_details = self.active_limit_entry_order_details
                                self._place_sl_tp_orders_after_entry(
                                    position_side=self.current_position_on_exchange, position_qty=self.current_position_quantity,
                                    entry_price=self.current_position_entry_price, sl_price_from_signal=order_details["sl_price"],
                                    tp_price_from_signal=order_details["tp_price"], price_precision=order_details["price_precision_asset"],
                                    tick_size=order_details["tick_size_str"]
                                )
                                
                                commission_str = order_data.get('n', '0')
                                if order_data.get('N') == 'USDT' and commission_str and Decimal(commission_str) > 0:
                                    old_capital = self.virtual_bot_capital_usdt
                                    self.virtual_bot_capital_usdt -= Decimal(commission_str)
                                    self.logger.info(f"Entry commission {Decimal(commission_str)} USDT. Virtual Capital: {old_capital:.4f} -> {self.virtual_bot_capital_usdt:.4f} USDT")

                        if order_status_ws == 'FILLED':
                            self.logger.info(f"Limit entry order {order_id_ws} is FULLY FILLED.")
                            self._reset_active_limit_entry_order_details() 

                    elif order_status_ws in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        self.logger.warning(f"Active limit entry order {order_id_ws} is {order_status_ws}.")
                        self._reset_active_limit_entry_order_details()

                elif order_status_ws == 'FILLED' and order_type_ws in ['STOP_MARKET', 'TAKE_PROFIT_MARKET', 'MARKET']:
                    if self.current_position_on_exchange is not None and not self.is_processing_position_close:
                        self.is_processing_position_close = True
                        action_type = "UNKNOWN CLOSE"
                        if order_type_ws == 'TAKE_PROFIT_MARKET':
                            action_type = 'TAKE PROFIT'
                            self.active_tp_order_id = None
                        elif order_type_ws == 'STOP_MARKET':
                            action_type = 'STOP LOSS'
                            self.active_sl_order_id = None
                        elif order_type_ws == 'MARKET':
                            action_type = 'MARKET CLOSE'

                        self.logger.info(f"!!! {action_type} order {order_id_ws} FILLED !!! Позиція для {self.symbol} закрита.")
                        
                        capital_before_update = self.virtual_bot_capital_usdt
                        try:
                            realized_profit = Decimal(order_data.get('rp', '0'))
                            commission = Decimal(order_data.get('n', '0'))
                            net_pnl = realized_profit - commission
                            self.virtual_bot_capital_usdt += net_pnl
                            self.logger.info(
                                f"COMPOUNDING: Gross PnL: {realized_profit:+.4f}, Commission: {commission:+.4f}, Net PnL: {net_pnl:+.4f}. "
                                f"Virtual Capital: {capital_before_update:.4f} -> {self.virtual_bot_capital_usdt:.4f} USDT."
                            )
                            self._check_capital_and_stop_if_needed("Просадка капіталу після закриття угоди.")
                        except Exception as e:
                            self.logger.error(f"Помилка оновлення капіталу: {e}", exc_info=True)
                        
                        self._cancel_all_orders_for_symbol()
                        self._reset_position_state()

            elif event_type == 'ACCOUNT_UPDATE':
                account_data = msg_data.get('a', {})
                for pos_update in account_data.get('P', []):
                    if pos_update.get('s') == self.symbol:
                        pos_amt = float(pos_update.get('pa', '0'))

                        if pos_amt == 0 and self.current_position_on_exchange is not None:
                            self.logger.info(f"ACCOUNT_UPDATE for {self.symbol} shows position is now zero. Awaiting final FILLED event.")
                        
                        elif pos_amt != 0 and self.current_position_on_exchange is None and self.active_limit_entry_order_details.get("orderId") is None:
                            entry_price = Decimal(pos_update.get('ep', '0'))
                            position_side = 'LONG' if pos_amt > 0 else 'SHORT'
                            if entry_price > 0:
                                self.current_position_on_exchange = position_side
                                self.current_position_quantity = abs(pos_amt)
                                self.current_position_entry_price = entry_price
                                self.active_sl_order_id = "EXTERNAL"
                                self.active_tp_order_id = "EXTERNAL"
                                self.logger.critical(
                                    f"КРИТИЧНА СИНХРОНІЗАЦІЯ: Виявлено зовнішню позицію {position_side} "
                                    f"({self.current_position_quantity} @ {entry_price})! Бот синхронізував свій стан "
                                    "і блокує нові угоди."
                                )

        except Exception as e:
            self.logger.error(f"Error in _on_websocket_message for {self.symbol}: {e}. Message: {message}", exc_info=True)
            
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
            self.logger.info(f"ML модель успішно завантажена з файлу: {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Помилка завантаження моделі: {e}")
            return None
    

# Всередині класу TradingBotInstance
    def _calculate_indicators(self, df_main: pd.DataFrame):
        if df_main.empty: return pd.DataFrame()
        
        df = df_main.copy().astype(np.float32)

        # 1. БАЗОВІ
        df.ta.atr(length=14, append=True)
        df['atr'] = df['ATRr_14'].fillna(method='bfill')
        df['vol_ma'] = df['volume'].rolling(window=20).mean().fillna(method='bfill')

        # 2. ОТРИМУЄМО 9 ФІКСОВАНИХ СІТОК
        # Наприклад: [10, 25, 50, ..., 2500]
        asset_grid = get_fixed_asset_grid(df, FULL_STEP_GRID, num_neighbors=4)
        self.current_asset_grid = asset_grid # Зберігаємо для інфо

        # Підготовка масивів для Numba
        closes = df['close'].values.astype(np.float64)
        
        # 3. ЦИКЛ ПО ВСІХ СІТКАХ (МУЛЬТИМАСШТАБ)
        # i - це індекс від 0 до 8 (якщо 9 сіток)
        for i, step_val in enumerate(asset_grid):
            # Рахуємо фізику для конкретного step_val
            vel, dist = calculate_single_step_physics_numba(closes, float(step_val))
            
            # Зберігаємо "сирі" числа
            # (Можна не зберігати в DF, щоб економити пам'ять, а зразу токенізувати, 
            # але для дебагу хай будуть)
            col_dist = f'dist_{i}'  
            col_vel = f'vel_{i}'
            
            df[col_dist] = dist.astype(np.float32)
            df[col_vel] = vel.astype(np.float32)
            
            # --- ТОКЕНІЗАЦІЯ ДЛЯ ЦЬОГО ГРІДА ---
            # Token Zone
            df[f't_zone_{i}'] = np.digitize(dist, MARKET_VOCAB['zone_bins']) - 1
            
            # Token Speed (Згладжуємо velocity)
            # Використовуємо pd.Series для rolling
            smooth_vel = pd.Series(vel).rolling(3).mean().fillna(0)
            df[f't_speed_{i}'] = np.digitize(smooth_vel, MARKET_VOCAB['speed_bins']) - 1

        # 4. ГЛОБАЛЬНІ ФІЧІ (НЕ ЗАЛЕЖАТЬ ВІД СІТКИ)
        # Effort (Зусилля) - воно одне на всі сітки
        df['effort'] = (df['volume'] / df['vol_ma']).fillna(0).astype(np.float32)
        df['t_vol'] = np.digitize(df['effort'], MARKET_VOCAB['vol_bins']) - 1
        
        # Активний (Центральний) Степ - просто для інформації боту (для розрахунку TP/SL)
        # Беремо середній елемент нашої сітки (ми ж центрували її по ATR)
        center_step_val = asset_grid[len(asset_grid)//2] 
        df['active_step'] = float(center_step_val)

        # Result (Глобальний рух відносно ATR)
        # Його теж можна лишити одного
        change = df['close'].diff().abs()
        df['result'] = (change / df['atr']).fillna(0).astype(np.float32)
        df['absorption'] = (df['effort'] / (df['result'] + 0.001)).astype(np.float32)
        
        df.fillna(0, inplace=True)
        return df
    
# ДОДАЙТЕ ЦЮ НОВУ ФУНКЦІЮ В КЛАС TradingBotInstance
    
    def _calculate_fractal_and_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ВИПРАВЛЕНА ВЕРСІЯ. Додано розрахунок фрактальних індексів.
        """
        self.logger.info("Calculating fractal and VSA features...")
        
        if 'atr_tf' not in df.columns or 'close_tf' not in df.columns:
            self.logger.error("Fractal features require 'atr_tf' and 'close_tf'. Calculation skipped.")
            return df

        coeffs = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        fractal_step_levels = {}
        
        base_percent_tf = (df['atr_tf'] / df['close_tf'].replace(0, np.nan)) * 100

        for coeff in coeffs:
            percent_tf = base_percent_tf * coeff
            idx = (percent_tf * df['close'] / 100).ffill().replace(0, 1e-10)
            
            valid_idx_mask = (idx > 0) & pd.notna(idx)
            magnitude = pd.Series(np.nan, index=df.index)
            if valid_idx_mask.any():
                magnitude.loc[valid_idx_mask] = 10 ** np.floor(np.log10(idx[valid_idx_mask]))
            magnitude.ffill(inplace=True)
            magnitude.fillna(1e-10, inplace=True)

            residual = (idx / magnitude.replace(0, 1e-10)).ffill()
            smooth_residual = residual.ewm(alpha=(1/self.residual_smooth), adjust=False).mean().ffill()
            
            conditions = [smooth_residual < 1.5, (smooth_residual >= 1.5) & (residual < 3.5), (smooth_residual >= 1.5) & (residual >= 3.5) & (residual < 7.5)]
            choices = [magnitude * 1, magnitude * 2, magnitude * 5]
            x_coeff = np.select(conditions, choices, default=magnitude * 10)
            # # --- ПОЧАТОК ВИПРАВЛЕННЯ: Застосовуємо той самий коефіцієнт 0.25 ---
            # x_coeff = x_coeff * 0.25
            # # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---
            df[f'x_{coeff}'] = pd.Series(x_coeff, index=df.index).ffill()
            
            rounded_close = np.round(df['close'] / df[f'x_{coeff}'].replace(0, 1e-10)) * df[f'x_{coeff}']
            step_level_col = f'step_level_{coeff}'
            df[step_level_col] = calculate_step_level_numba(df['close'].to_numpy(), rounded_close.to_numpy(), df[f'x_{coeff}'].to_numpy())
            df[step_level_col].ffill(inplace=True)
            fractal_step_levels[coeff] = df[step_level_col]

            # --- ★★★ ПОЧАТОК ВИДАЛЕННЯ (v5) ★★★ ---
            # !!! ОСНОВНЕ ВИПРАВЛЕННЯ: Додаємо розрахунок індексу для фрактального степу !!!
            # df[f'fractal_step_{coeff}_index'] = calculate_step_index(df[step_level_col].to_numpy())
            # --- ★★★ КІНЕЦЬ ВИДАЛЕННЯ (v5) ★★★ ---

        # --- Розрахунок похідних фіч (без змін) ---
        step_diffs = pd.concat([fractal_step_levels[c].diff() for c in coeffs if c in fractal_step_levels], axis=1)
        step_signs = step_diffs.apply(np.sign)
        df['Fractal_Alignment_Score'] = step_signs.sum(axis=1)

        for coeff in coeffs:
            step_group_id = (df[f'step_level_{coeff}'] != df[f'step_level_{coeff}'].shift()).cumsum()
            duration = df.groupby(step_group_id).cumcount() + 1
            step_size = df[f'step_level_{coeff}'].diff().abs()
            df[f'Velocity_{coeff}'] = (step_size / duration.shift(1)).fillna(0)

        df['Micro_to_Macro_Ratio'] = (df['x_0.5'] / df['x_2.0']).fillna(1)
        
        base_step_level = df['step_level_1.0']
        base_step_group_id = (base_step_level != base_step_level.shift()).cumsum()
        
        is_breakout = base_step_level != base_step_level.shift()
        mean_volume_in_step = df.groupby(base_step_group_id)['volume'].transform('mean')
        df['Step_Breakout_Volume_Spike'] = np.where(is_breakout, (df['volume'] / mean_volume_in_step.shift(1)).fillna(1.0), 1.0)
        
        is_in_retracement = (df['close'] < base_step_level) | (df['close'] > base_step_level)
        rolling_vol_mean = df['volume'].rolling(window=5, min_periods=1).mean()
        df['Retracement_Volume_Decline'] = np.where(is_in_retracement, (df['volume'] / rolling_vol_mean).fillna(1.0), 1.0)

        body_size = (df['close'] - df['open']).abs()
        volume_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
        distance_to_step = (df['close'] - base_step_level).abs()
        x_1_0_numeric = pd.to_numeric(df['x_1.0'])
        is_absorption = (body_size < body_size.quantile(0.2)) & (volume_ratio > volume_ratio.quantile(0.8)) & (distance_to_step < (x_1_0_numeric * 0.1))
        df['Absorption_Volume_Cluster'] = is_absorption.astype(int)

        # --- ★★★ ПОЧАТОК ВИДАЛЕННЯ (v5) ★★★ ---
        # --- ПОЧАТОК НОВОГО БЛОКУ: "Макро-Сітка" (Твоя Ідея) ---
        # self.logger.info("Розрахунок нових 'Макро-Сітки' 3D-координат...")
        
        # # 1. Визначаємо базові рівні
        # # "Нуль" = Макро-степ (4.0)
        # zero_level = df['step_level_4.0']
        # # "Розмір" = x від Макро-степу (4.0)
        # grid_size = df['x_4.0'].replace(0, np.nan) # Захист від ділення на нуль
        # # "Позиція" = наш основний (мікро) степ
        # current_level = df['step_level'] 
        
        # # 2. Розраховуємо індекс "Макро-Сітки" за твоєю формулою
        # # (Поточний Мікро-Степ - Поточний Макро-Степ) / Розмір Макро-Сітки
        # df['macro_grid_index'] = np.floor((current_level - zero_level) / grid_size)
        
        # # 3. Розраховуємо індекс попередньої комірки
        # df['prev_macro_grid_index'] = df['macro_grid_index'].shift(1)
        
        # # 4. Очистка: Заповнюємо NaN, що виникли на початку або через ділення на нуль
        # # ffill() гарантує, що ми залишаємось "в комірці", поки не перейдемо в нову
        # df['macro_grid_index'].ffill(inplace=True)
        # df['prev_macro_grid_index'].ffill(inplace=True)
        
        # # Заповнюємо залишкові NaN (на самому початку) нулем і робимо цілим числом
        # df['macro_grid_index'].fillna(0, inplace=True)
        # df['prev_macro_grid_index'].fillna(0, inplace=True)
        
        # df['macro_grid_index'] = df['macro_grid_index'].astype(int)
        # df['prev_macro_grid_index'] = df['prev_macro_grid_index'].astype(int)
        
        # --- КІНЕЦЬ НОВОГО БЛОКУ ---
        # --- ★★★ КІНЕЦЬ ВИДАЛЕННЯ (v5) ★★★ ---

        self.logger.info("Fractal and VSA features calculated successfully.")
        return df
    

    def _check_and_execute_trade_logic(self, df_with_indicators: pd.DataFrame):
        if self.is_processing_position_close:
            return

        # --- БЛОК 0: СЕРВІСНІ ПЕРЕВІРКИ ---
        if self._check_capital_and_stop_if_needed("Перевірка капіталу."):
            return
        
        # Скасування застарілого лімітного ордера
        if self.active_limit_entry_order_details.get("orderId"):
            order_details = self.active_limit_entry_order_details
            last_candle = df_with_indicators.iloc[-1]
            current_step_val_check = Decimal(str(last_candle['step_level'])) # step_level still exists from _calculate_indicators logic if you kept it, but maybe rely on active_step now? 
            # CAUTION: If you removed step_level calculation, use active_step logic or re-verify. 
            # Assuming 'step_level' logic was preserved or mapped to 'active_step' logic.
            # Let's assume standard step_level logic is still there or active_step is the new standard.
            # If using new logic strictly, this might need adjustment to active_step.
            # For safety, let's assume step_level is calculated in _calculate_indicators as before or we check against active_step.
            step_level_at_signal = order_details.get("step_level_at_signal")
            
            # If strictly using new active_step logic, you might want to check if the 'active_step' changed significantly or price moved away.
            # But let's keep original logic if step_level is still in DF.
            if 'step_level' in last_candle and current_step_val_check != step_level_at_signal:
                 self.logger.warning(f"Скасовую застарілий ордер ID {order_details.get('orderId')}, оскільки степ змінився.")
                 if self._cancel_order(order_details.get("orderId")):
                     self._reset_active_limit_entry_order_details()

        # --- БЛОК 1: ДЕТЕКТОР ЗЛАМУ СТЕПУ ---
        # Перевіряємо, чи достатньо даних для формування послідовності
        if len(df_with_indicators) < TIMESTEPS + 5: 
            return
            
        last_row = df_with_indicators.iloc[-1]
        prev_row = df_with_indicators.iloc[-2]
        
        # Using active_step logic often replaces step_level for breakout detection in dynamic grids
        # But if you kept step_level for compatibility, use it. 
        # If new logic relies purely on active_step (volatility), you might trigger on 'active_step' change 
        # OR simply on every candle where conditions are met.
        # "Value Entry" logic often implies entering on a retracement to a level.
        # Let's assume we proceed if we have data.
        
        # --- БЛОК 2: ЗВІТНІСТЬ ---
        if self.pending_prediction_for_report:
            self.pending_prediction_for_report = None

        # --- БЛОК 3: ПРОГНОЗУВАННЯ (HYBRID TRANSFORMER) ---
        if self.current_position_on_exchange or self.active_limit_entry_order_details.get("orderId"):
            self.logger.info("Пропуск нового прогнозу: є активна позиція або ордер.")
            return

        try:
            # 1. Підготовка даних (Sequence Generation)
            # Беремо останні TIMESTEPS свічок
            # Важливо: create_hybrid_dataset очікує DF. 
            # Ми передаємо слайс, який включає TIMESTEPS рядків.
            
            # loc_prev points to the row BEFORE the current live candle (last closed candle)
            # In live trading, we usually predict on the just-closed candle sequence.
            
            sequence_df = df_with_indicators.iloc[-TIMESTEPS:]
            
            if len(sequence_df) != TIMESTEPS:
                self.logger.warning(f"Недостатньо історії для Трансформера (маємо {len(sequence_df)}, треба {TIMESTEPS}).")
                return

            if sequence_df.isnull().values.any():
                self.logger.warning("У вхідних даних знайдено NaN. Пропуск прогнозу.")
                # Optional: self.logger.debug(f"NaN columns: {sequence_df.columns[sequence_df.isna().any()].tolist()}")
                return

            # 2. Масштабування та Форматування (Використовуємо Helper)
            if self.scaler:
                # is_training=False використовує збережений скейлер
                # X_list буде списком [X_zone, X_speed, X_vol, X_floats]
                # Але create_hybrid_dataset робить вікна (N, T, F). 
                # Для одного прогнозу N=1.
                X_list, _ = create_hybrid_dataset(sequence_df, scaler=self.scaler, is_training=False)
            else:
                self.logger.error("Скейлер не завантажено! Прогноз неможливий.")
                return
            
            # 3. Прогноз
            if self.model:
                # X_list вже має потрібну форму списку масивів (1, 48, ...)
                prediction_prob = self.model.predict(X_list, verbose=0)[0][0]
            else:
                self.logger.error("Модель Transformer не завантажена! Прогноз неможливий.")
                return
            
            self.logger.info(f"HYBRID PREDICTION: Probability = {prediction_prob:.4f}")

            # 4. Прийняття рішення (Поріг 0.5)
            predicted_class = 1 if prediction_prob >= 0.5 else 0
            # У Value Entry Sniper ми прогнозуємо успіх входу.
            # Але який напрямок?
            # У навчанні ми розмічали: (close.shift(-5) > close) -> 1 (UP)
            # Тобто 1 = Price will go UP.
            entry_side = 'LONG' if predicted_class == 1 else 'SHORT'
            
            # Фільтр впевненості (можна додати)
            if prediction_prob < 0.55 and prediction_prob > 0.45:
                 return # Не впевнені

            self.logger.critical(f"!!! СИГНАЛ TRANSFORMER: {entry_side} (Prob: {prediction_prob:.4f}) !!!")

            # --- БЛОК 4: ВИКОНАННЯ УГОДИ ---
            # Використовуємо active_step (динамічний) замість x (статичного)
            # active_step вже розрахований в _calculate_indicators
            current_step_size = Decimal(str(last_row['active_step']))
            current_close = Decimal(str(last_row['close']))
            
            if current_step_size == 0: return
            
            is_long_trade = (entry_side == 'LONG')
            
            # Логіка Снайпера: вхід на відкаті
            # Ми хочемо увійти, коли ціна "дихає" проти тренду, щоб отримати кращу ціну
            # Наприклад, входимо ліміткою трохи краще поточної ціни
            
            retracement_factor = Decimal('0.2') # 20% від кроку - невеликий відкат
            
            if is_long_trade:
                # Хочемо купити дешевше
                entry_price = current_close - (current_step_size * retracement_factor)
                tp_price = entry_price + (current_step_size * Decimal('1.5'))
                sl_price = entry_price - (current_step_size * Decimal('0.65'))
            else:
                # Хочемо продати дорожче
                entry_price = current_close + (current_step_size * retracement_factor)
                tp_price = entry_price - (current_step_size * Decimal('1.5'))
                sl_price = entry_price + (current_step_size * Decimal('0.65'))
            
            self.logger.warning(f"ПАРАМЕТРИ УГОДИ: {entry_side} for {self.symbol}. Вхід: {entry_price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")
            
            # --- РИЗИК МЕНЕДЖМЕНТ (Стандартний) ---
            target_risk_pct = Decimal(str(self.target_percent_param)) / Decimal('100')
            price_move_to_sl_abs = abs(entry_price - sl_price)
            
            if entry_price <= 0 or price_move_to_sl_abs <= 0: return
            
            price_move_to_sl_pct = (price_move_to_sl_abs / entry_price)
            if price_move_to_sl_pct <= 0: return
            
            required_leverage = int(target_risk_pct / price_move_to_sl_pct)
            if required_leverage < 1: required_leverage = 1
            
            notional_value_check = float(self.virtual_bot_capital_usdt * Decimal(str(required_leverage)))
            max_lev_allowed = self._get_max_allowed_leverage_for_notional(notional_value_check)
            final_leverage = min(required_leverage, max_lev_allowed)
            if final_leverage < 1: final_leverage = 1
            
            self.client.change_leverage(symbol=self.symbol, leverage=final_leverage)

            position_notional = (self.virtual_bot_capital_usdt * Decimal(str(final_leverage))) * Decimal('0.98')
            quantity_dec = position_notional / entry_price
            quantity = self._round_quantity_to_step_size(float(quantity_dec), self.symbol_trading_rules.get('stepSize'))
            
            min_notional = Decimal(str(self.symbol_trading_rules.get('minNotional', '5.0')))
            if (Decimal(str(quantity)) * entry_price) < min_notional:
                self.logger.warning(f"Розрахована кількість {quantity} замала для мін. номіналу. Пропуск.")
                return

            tick_size_str = self.symbol_trading_rules.get('tickSize')
            final_entry_price = self._round_price_to_tick_size(entry_price, tick_size_str, side="TP_SHORT" if is_long_trade else "TP_LONG")
            
            order_action_side = 'BUY' if is_long_trade else 'SELL'
            new_client_order_id = f"x-tf-{int(time.time() * 1000)}"
            
            # Відправка лімітного ордера
            entry_limit_order_response = self._place_limit_order(
                side=order_action_side, quantity=quantity, price=float(final_entry_price),
                position_side=entry_side, new_client_order_id=new_client_order_id
            )
            
            if entry_limit_order_response and entry_limit_order_response.get('orderId'):
                self.logger.info(f"ACTION: New Entry LIMIT order request sent. OrderID: {entry_limit_order_response['orderId']}.")
                self.active_limit_entry_order_details.update({
                    "orderId": str(entry_limit_order_response['orderId']), "symbol": self.symbol,
                    "clientOrderId": new_client_order_id, "order_side": order_action_side, "position_side": entry_side,
                    "price": final_entry_price, "sl_price": sl_price, "tp_price": tp_price, "status": 'NEW',
                    "price_precision_asset": self.symbol_trading_rules.get('pricePrecision'),
                    "tick_size_str": self.symbol_trading_rules.get('tickSize'), 
                    # Use current_close or a specific level as 'step_level_at_signal' reference if needed
                    "step_level_at_signal": current_close 
                })

        except Exception as e:
            self.logger.error(f"Помилка в блоці прогнозування та торгівлі (Transformer): {e}", exc_info=True)
            self.pending_prediction_for_report = None


    def _place_limit_order(self, side: str, quantity: float, price: float, position_side: str, new_client_order_id: str):
        if quantity <= 0 or price <= 0:
            self.logger.warning(f"Attempted to place LIMIT order for {self.symbol} with invalid quantity {quantity} or price {price}. Order not placed.")
            return None

        price_precision = self.symbol_trading_rules.get('pricePrecision', 2) # Отримуємо точність ціни з правил
        # Точність кількості вже має бути врахована в _calculate_quantity
        formatted_price = f"{price:.{price_precision}f}"


        self.logger.info(f"Attempting to place LIMIT order for {self.symbol}: {side} {quantity} at price {formatted_price}, positionSide: {position_side}")
        try:
            order_params = {
                'symbol': self.symbol, # Використовуємо self.symbol
                'side': side,
                'positionSide': position_side,
                'type': 'LIMIT',
                'quantity': quantity, # Кількість вже має бути правильно округлена
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

    def _cancel_order(self, order_id_str: str):
        if not order_id_str:
            self.logger.warning(f"_cancel_order called with no order_id_str for {self.symbol}.")
            return False # Повертаємо False, оскільки операція не виконана
        try:
            order_id_to_cancel = int(order_id_str)
            self.logger.info(f"Attempting to cancel order ID {order_id_to_cancel} for {self.symbol}...")
            self.client.cancel_order(symbol=self.symbol, orderId=order_id_to_cancel) # Використовуємо self.symbol
            self.logger.info(f"Order ID {order_id_to_cancel} for {self.symbol} cancelled successfully.")
            return True
        except ClientError as e:
            if e.error_code == -2011: 
                self.logger.warning(f"Could not cancel order ID {order_id_str} for {self.symbol}: Order already filled/canceled or does not exist. (Code: {e.error_code})")
                return True # Вважаємо успіхом, бо ордера вже немає або він неактивний
            self.logger.error(f"API Error cancelling order ID {order_id_str} for {self.symbol}: {e.error_message} (Code: {e.error_code})")
        except ValueError:
            self.logger.error(f"Invalid order_id format for cancellation ({self.symbol}): '{order_id_str}'. Cannot convert to int.")
        except Exception as e:
            self.logger.error(f"Unexpected error cancelling order ID {order_id_str} for {self.symbol}: {e}", exc_info=True)
        return False # Повертаємо False у разі помилки

    # Додай цей НОВИЙ метод у клас TradingBotInstance

    def _cancel_all_orders_for_symbol(self, retries: int = 3, delay: int = 2):
        """Скасовує ВСІ відкриті ордери для поточного символу."""
        self.logger.info(f"Attempting to cancel ALL open orders for {self.symbol}...")
        for attempt in range(retries):
            try:
                self.client.cancel_open_orders(symbol=self.symbol)
                self.logger.info(f"Successfully sent request to cancel all open orders for {self.symbol}.")
                return True
            except ClientError as e:
                # Код -2011 означає "Unknown order sent", що трапляється, якщо ордерів вже немає.
                # Це не помилка, а успішний результат.
                if e.error_code == -2011:
                    self.logger.info(f"No open orders found for {self.symbol} to cancel, which is a success.")
                    return True
                self.logger.error(f"API Error cancelling all orders for {self.symbol} (Attempt {attempt + 1}/{retries}): {e.error_message}")
            except Exception as e:
                self.logger.error(f"Unexpected error cancelling all orders for {self.symbol} (Attempt {attempt + 1}/{retries}): {e}", exc_info=True)
            
            if attempt < retries - 1:
                time.sleep(delay)
        
        self.logger.critical(f"CRITICAL: Failed to cancel all orders for {self.symbol} after {retries} attempts.")
        return False
    
    def _round_price_to_tick_size(self, price, tick_size_str: str, side: str = "DEFAULT"):
        # ВИПРАВЛЕНО: Функція тепер завжди повертає Decimal для збереження точності
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
             # Для звичайного округлення ціни ордера використовуємо ROUND_DOWN для покупок і ROUND_UP для продажів, але тут ми не знаємо напрямку, тому краще не використовувати DEFAULT
             self.logger.warning(f"Rounding price with DEFAULT side for {self.symbol}. This may lead to suboptimal execution. Please specify rounding side.")
             rounding_mode = ROUND_DOWN # Безпечне значення за замовчуванням
        
        rounded_price_dec = (price_dec / tick_dec).to_integral_value(rounding=rounding_mode) * tick_dec
        
        # Повертаємо Decimal, а не float
        return rounded_price_dec.quantize(Decimal('1e-' + str(num_decimals)))
    

    def _ensure_all_orders_canceled_for_shutdown(self):
        self.logger.info(f"Перевірка та скасування активних ордерів для {self.symbol} під час зупинки...")
        
        if self.active_limit_entry_order_details and self.active_limit_entry_order_details.get("orderId"):
            limit_order_id = self.active_limit_entry_order_details.get("orderId")
            status = self.active_limit_entry_order_details.get("status")
            if status not in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED', 'PENDING_CANCEL']:
                self.logger.info(f"Спроба скасувати активний лімітний ордер на вхід {limit_order_id} для {self.symbol}.")
                self._cancel_order(limit_order_id) 
            else:
                self.logger.info(f"Лімітний ордер на вхід {limit_order_id} для {self.symbol} вже в стані {status} або очікує скасування.")

        if self.active_sl_order_id:
            self.logger.info(f"Спроба скасувати активний SL ордер {self.active_sl_order_id} для {self.symbol}.")
            if self._cancel_order(self.active_sl_order_id):
                self.active_sl_order_id = None

        if self.active_tp_order_id:
            self.logger.info(f"Спроба скасувати активний TP ордер {self.active_tp_order_id} для {self.symbol}.")
            if self._cancel_order(self.active_tp_order_id):
                self.active_tp_order_id = None
        
        self.logger.info(f"Перевірка активних ордерів для {self.symbol} завершена.")
        time.sleep(1.0)

    def _close_open_position_on_shutdown(self):
        """Перевіряє наявність відкритої позиції і закриває її ринковим ордером при зупинці."""
        self.logger.info(f"Перевірка наявності відкритої позиції для {self.symbol} перед зупинкою...")
        
        try:
            # Робимо свіжий запит до біржі, щоб отримати найактуальнішу інформацію
            positions = self._get_open_positions(symbol_to_check=self.symbol, retries=1)
            
            if positions: # _get_open_positions повертає список
                position_info = positions[0]
                position_amt = float(position_info.get('positionAmt', 0))
                
                if position_amt != 0:
                    self.logger.warning(
                        f"Виявлено відкриту позицію: {position_amt} {self.symbol}. "
                        f"Ініціюю примусове закриття по ринку."
                    )
                    
                    # Спочатку скасовуємо всі лімітні ордери (SL/TP)
                    self._ensure_all_orders_canceled_for_shutdown()
                    time.sleep(1) # Невелика пауза

                    # Визначаємо сторону для закриття
                    closing_side = "SELL" if position_amt > 0 else "BUY"
                    position_side_to_close = "LONG" if position_amt > 0 else "SHORT"
                    
                    # Виставляємо ринковий ордер на закриття
                    close_order = self._place_market_order_to_close(
                        order_closing_side=closing_side,
                        quantity=abs(position_amt),
                        position_being_closed_side=position_side_to_close
                    )
                    
                    if close_order and close_order.get('orderId'):
                        self.logger.info(f"Ринковий ордер на закриття позиції {self.symbol} успішно відправлено.")
                        time.sleep(3) # Даємо час на виконання
                    else:
                        self.logger.error(f"НЕ ВДАЛОСЯ закрити позицію для {self.symbol} при зупинці! ПОТРІБНЕ РУЧНЕ ВТРУЧАННЯ!")
                else:
                    self.logger.info(f"Активних позицій для {self.symbol} на біржі не виявлено.")
            else:
                self.logger.info(f"Не вдалося отримати інформацію про позиції або їх немає.")

        except Exception as e:
            self.logger.error(f"Помилка під час примусового закриття позиції для {self.symbol}: {e}", exc_info=True)

    def _calculate_quantity(self, price, capital, leverage,
                           step_size_str: str, min_qty_val: float, qty_precision_from_rules: int):
        # ВИПРАВЛЕНО: Конвертуємо вхідні дані в Decimal для безпечних розрахунків
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

            # _round_quantity_to_step_size вже повертає float, це коректно
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
        # Округлення вниз до найближчого кратного step_size
        rounded_qty_dec = (quantity_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
        
        # Визначення кількості знаків після коми з step_size_str
        if '.' in step_size_str:
            num_decimals = len(step_size_str.split('.')[1])
        else:
            num_decimals = 0
            
        return float(rounded_qty_dec.quantize(Decimal('1e-' + str(num_decimals))))


    def _get_max_allowed_leverage_for_notional(self, notional_value: float):
        if not self.leverage_brackets_for_symbol:
            self.logger.warning(f"Leverage bracket data for {self.symbol} is not available. Returning default max leverage (e.g., 20x).")
            return 20

        sorted_brackets = sorted(self.leverage_brackets_for_symbol, key=lambda x: float(x.get('notionalFloor', 0))) # Додано .get
        
        # Проходимо від найбільших номінальних значень до менших
        for bracket in reversed(sorted_brackets):
            floor = float(bracket.get('notionalFloor', 0))
            # cap = float(bracket.get('notionalCap', float('inf'))) # Верхня межа брекету
            # Якщо номінальна вартість більша або дорівнює нижній межі цього брекету,
            # то це плече для даного брекету
            if notional_value >= floor:
                return int(bracket.get('initialLeverage', 1)) # Повертаємо плече для цього брекету

        # Якщо номінальна вартість менша за всі notionalFloor (малоймовірно, якщо є брекет з floor=0)
        # або якщо список брекетів порожній після сортування (теж малоймовірно, якщо _get_and_store_leverage_data працює)
        if sorted_brackets: # Якщо список не порожній, беремо плече з найменшого брекету (першого після сортування)
             self.logger.warning(f"Could not determine specific leverage bracket for notional {notional_value} for {self.symbol}. Using leverage from the smallest notional bracket.")
             return int(sorted_brackets[0].get('initialLeverage', 1))
        
        self.logger.warning(f"Could not determine appropriate leverage bracket for notional value {notional_value} USDT for {self.symbol}. Returning 1x for safety.")
        return 1

    def _place_sl_tp_orders_after_entry(self, position_side: str, position_qty: float,
                                        entry_price: Decimal,
                                        sl_price_from_signal: Decimal, # <--- ВИКОРИСТОВУЄМО ПЕРЕДАНІ ЗНАЧЕННЯ
                                        tp_price_from_signal: Decimal, # <--- ВИКОРИСТОВУЄМО ПЕРЕДАНІ ЗНАЧЕННЯ
                                        price_precision: int, tick_size: str,
                                        liquidation_price: Decimal = Decimal('0')):
        
        self.logger.info(f"Placing SL/TP for {position_side} {position_qty} {self.symbol} @ {entry_price}")
        self.logger.info(f"  -> Using pre-calculated levels: SL={sl_price_from_signal}, TP={tp_price_from_signal}")

        # Прибираємо весь старий блок розрахунку з множниками, бо ціни вже відомі
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
            self.logger.error(f"Invalid position_side '{position_side}' for SL/TP on {self.symbol}.")
            return

        # Округлення та перевірки залишаються, але тепер вони працюють з правильними цілями
        sl_price_final = self._round_price_to_tick_size(sl_price_raw, tick_size, side=rounding_sl_side)
        tp_price_final = self._round_price_to_tick_size(tp_price_raw, tick_size, side=rounding_tp_side)
        self.logger.info(f"Calculated SL price: {sl_price_final:.{price_precision}f}, TP price: {tp_price_final:.{price_precision}f} for {self.symbol}")
        
        sl_is_valid = True
        sl_invalidation_reason = ""
        tick_size_decimal = Decimal(tick_size) if tick_size else Decimal('0')
        min_price_distance = self.min_distance_from_entry_ticks * tick_size_decimal

        # --- Перевірка валідності Stop Loss ---
        if sl_price_final > 0 and liquidation_price > 0 and tick_size_decimal > 0:
            buffer_value = self.liquidation_price_buffer_ticks * tick_size_decimal
            original_sl_before_liq_adj = sl_price_final
            if position_side == 'LONG' and sl_price_final <= (liquidation_price + buffer_value):
                sl_price_final = self._round_price_to_tick_size(liquidation_price + buffer_value, tick_size, side=rounding_sl_side)
                self.logger.info(f"SL price for {self.symbol} changed from {original_sl_before_liq_adj:.{price_precision}f} to {sl_price_final:.{price_precision}f} after liquidation check.")
            elif position_side == 'SHORT' and sl_price_final >= (liquidation_price - buffer_value):
                sl_price_final = self._round_price_to_tick_size(liquidation_price - buffer_value, tick_size, side=rounding_sl_side)
                self.logger.info(f"SL price for {self.symbol} changed from {original_sl_before_liq_adj:.{price_precision}f} to {sl_price_final:.{price_precision}f} after liquidation check.")

        if position_side == 'LONG' and sl_price_final >= entry_price:
            sl_is_valid, sl_invalidation_reason = False, "SL price is on the wrong side of entry"
        elif position_side == 'SHORT' and sl_price_final <= entry_price:
            sl_is_valid, sl_invalidation_reason = False, "SL price is on the wrong side of entry"

        if sl_is_valid and sl_price_final > 0 and tick_size_decimal > 0:
            if abs(entry_price - sl_price_final) < min_price_distance:
                sl_is_valid, sl_invalidation_reason = False, f"SL price {sl_price_final} too close to entry {entry_price}"
        
        if sl_is_valid and sl_price_final > 0:
            potential_loss_on_sl_abs = abs(entry_price - sl_price_final) * Decimal(str(position_qty))
            max_allowed_risk_abs = self.virtual_bot_capital_usdt * self.max_risk_percent_of_per_trade_capital
            if potential_loss_on_sl_abs > max_allowed_risk_abs:
                sl_is_valid, sl_invalidation_reason = False, f"Potential loss {potential_loss_on_sl_abs:.4f} exceeds max risk {max_allowed_risk_abs:.4f}"
        
        # --- Виставлення Stop Loss ---
        if sl_is_valid and sl_price_final > 0 and position_qty > 0:
            sl_order = self._place_stop_market_order(sl_order_side, position_qty, sl_price_final, price_precision, position_side, rounding_side=rounding_sl_side)
            if sl_order and sl_order.get('orderId'):
                self.active_sl_order_id = str(sl_order['orderId'])
                self.logger.info(f"ACTION: SL order placed for {self.symbol}. ID: {self.active_sl_order_id}")
            else:
                self.logger.error(f"ACTION_FAILED: Failed to place SL for {self.symbol}. Position open without SL!")
                self._close_position_critically(reason=f"SL placement API failed ({self.symbol})")
                return
        else:
            final_reason = sl_invalidation_reason if sl_invalidation_reason else f"Initial SL parameters invalid (price:{sl_price_final}, qty:{position_qty})"
            self.logger.error(f"ACTION_FAILED: SL is invalid for {self.symbol}. Reason: {final_reason}")
            self._close_position_critically(reason=final_reason)
            return

        # --- Перевірка валідності Take Profit ---
        tp_is_valid = True
        if tp_price_final <= 0:
            tp_is_valid = False
        elif position_side == 'LONG' and tp_price_final <= entry_price:
            tp_is_valid = False
        elif position_side == 'SHORT' and tp_price_final >= entry_price:
            tp_is_valid = False
        
        if tp_is_valid and tick_size_decimal > 0:
            if abs(tp_price_final - entry_price) < min_price_distance:
                tp_is_valid = False
                self.logger.warning(f"TP price {tp_price_final} for {self.symbol} too close to entry. TP not placed.")

        # --- Виставлення Take Profit ---
        if tp_is_valid and position_qty > 0:
            tp_order = self._place_take_profit_market_order(tp_order_side, position_qty, tp_price_final, price_precision, position_side, rounding_side=rounding_tp_side)
            if tp_order and tp_order.get('orderId'):
                self.active_tp_order_id = str(tp_order['orderId'])
                self.logger.info(f"ACTION: TP order placed for {self.symbol}. ID: {self.active_tp_order_id}")
            else:
                self.logger.warning(f"ACTION_FAILED: Failed to place TP order for {self.symbol}.")
        else:
            self.logger.warning(f"TP invalid or params invalid for {self.symbol}. TP not placed.")

            

    def _place_stop_market_order(self, side: str, quantity: float, stop_price: Decimal, price_precision: int, position_side_to_close: str, rounding_side: str): # <<< ВИПРАВЛЕНО: додано rounding_side
        if quantity <= 0 or stop_price <=0:
            self.logger.warning(f"Invalid parameters for STOP_MARKET order for {self.symbol}: quantity={quantity}, stop_price={stop_price}")
            return None
        try:
            # <<< ВИПРАВЛЕНО: Використовуємо переданий rounding_side замість 'DEFAULT'
            stop_price_rounded = self._round_price_to_tick_size(stop_price, self.symbol_trading_rules.get('tickSize', '0.01'), side=rounding_side)
            stop_price_formatted = f"{stop_price_rounded:.{price_precision}f}"

            self.logger.info(f"Attempting to place STOP_MARKET order for {self.symbol}: {side} {quantity} at stopPrice {stop_price_formatted}, closing {position_side_to_close}")
            order = self.client.new_order(
                symbol=self.symbol, side=side, positionSide=position_side_to_close,
                type="STOP_MARKET", quantity=quantity, stopPrice=stop_price_formatted,
                closePosition=True 
            )
            self.logger.info(f"STOP_MARKET order placed successfully for {self.symbol}: {order}")
            return order
        except ClientError as e:
            self.logger.error(f"API Error placing STOP_MARKET for {self.symbol} {side} {quantity} stopPrice {stop_price_formatted} posSide {position_side_to_close}: {e.error_message} (Code: {e.error_code})")
        except Exception as e:
            self.logger.error(f"Unexpected error placing STOP_MARKET for {self.symbol}: {e}", exc_info=True)
        return None

    def _place_take_profit_market_order(self, side: str, quantity: float, stop_price: Decimal, price_precision: int, position_side_to_close: str, rounding_side: str): # <<< ВИПРАВЛЕНО: додано rounding_side
        if quantity <= 0 or stop_price <=0:
            self.logger.warning(f"Invalid parameters for TAKE_PROFIT_MARKET for {self.symbol}: quantity={quantity}, stop_price={stop_price}")
            return None
        try:
            # <<< ВИПРАВЛЕНО: Використовуємо переданий rounding_side замість 'DEFAULT'
            stop_price_rounded = self._round_price_to_tick_size(stop_price, self.symbol_trading_rules.get('tickSize', '0.01'), side=rounding_side)
            stop_price_formatted = f"{stop_price_rounded:.{price_precision}f}"

            self.logger.info(f"Attempting to place TAKE_PROFIT_MARKET for {self.symbol}: {side} {quantity} at stopPrice {stop_price_formatted}, closing {position_side_to_close}")
            order = self.client.new_order(
                symbol=self.symbol, side=side, positionSide=position_side_to_close,
                type="TAKE_PROFIT_MARKET", quantity=quantity, stopPrice=stop_price_formatted,
                closePosition=True
            )
            self.logger.info(f"TAKE_PROFIT_MARKET order placed successfully for {self.symbol}: {order}")
            return order
        except ClientError as e:
            self.logger.error(f"API Error placing TAKE_PROFIT_MARKET for {self.symbol} {side} {quantity} stopPrice {stop_price_formatted} posSide {position_side_to_close}: {e.error_message} (Code: {e.error_code})")
        except Exception as e:
            self.logger.error(f"Unexpected error placing TAKE_PROFIT_MARKET for {self.symbol}: {e}", exc_info=True)
        return None

    def _place_market_order_to_close(self, order_closing_side: str, quantity: float, position_being_closed_side: str):
        if quantity <= 0:
            self.logger.warning(f"Attempted to place closing market order for {self.symbol} with quantity {quantity}. Not placed.")
            return None
        try:
            qty_precision_close = self.symbol_trading_rules.get('quantityPrecision', 3) # Беремо з правил
            # Кількість вже має бути правильно округлена до цього моменту
            # formatted_quantity_for_order = self._round_quantity_to_step_size(quantity, self.symbol_trading_rules.get('stepSize'))

            self.logger.info(f"Attempting to place MARKET CLOSE order for {self.symbol}: {order_closing_side} {quantity} to close {position_being_closed_side} position.")
            order_params = {
                'symbol': self.symbol, 
                'side': order_closing_side, 
                'positionSide': position_being_closed_side, 
                'type': 'MARKET', 
                'quantity': quantity, # Передаємо вже округлену кількість
            }
            order = self.client.new_order(**order_params)
            self.logger.info(f"MARKET CLOSE order request for {self.symbol} sent successfully. Response: {order}")
            return order
        except ClientError as e:
            self.logger.error(f"API ClientError placing MARKET CLOSE order for {self.symbol} {order_closing_side} {quantity} closing {position_being_closed_side}: "
                         f"Status {e.status_code}, Code {e.error_code}, Msg: {e.error_message}. Headers: {e.header}")
        except Exception as e:
            self.logger.error(f"Unexpected error placing MARKET CLOSE order for {self.symbol}: {e}", exc_info=True)
        return None


    def _close_position_critically(self, reason: str):
        if not self.current_position_on_exchange or self.current_position_quantity <= 0:
            self.logger.info(f"Critical close called for {self.symbol}, but no active position to close. Reason: {reason}")
            if not self.bot_stopped_due_to_total_sl: # Якщо бот ще не зупинений, зупиняємо
                self.bot_stopped_due_to_total_sl = True
                self.logger.critical(f"Bot {self.symbol} stopped. Reason: {reason} (no active position found).")
            return

        self.logger.critical(f"CRITICAL ACTION for {self.symbol}: Attempting to close position {self.current_position_on_exchange} ({self.current_position_quantity}) by MARKET due to: {reason}.")
        
        # Спочатку скасовуємо всі активні ордери для цього символу, пов'язані з поточною позицією
        if self.active_sl_order_id:
            self.logger.info(f"CritClose ({self.symbol}): Cancelling active SL order {self.active_sl_order_id}")
            self._cancel_order(self.active_sl_order_id)
            self.active_sl_order_id = None
        if self.active_tp_order_id:
            self.logger.info(f"CritClose ({self.symbol}): Cancelling active TP order {self.active_tp_order_id}")
            self._cancel_order(self.active_tp_order_id)
            self.active_tp_order_id = None
        # Також скасовуємо лімітний ордер на вхід, якщо він ще активний
        if self.active_limit_entry_order_details.get("orderId") and \
           self.active_limit_entry_order_details.get("status") not in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
            self.logger.info(f"CritClose ({self.symbol}): Cancelling pending limit entry order {self.active_limit_entry_order_details['orderId']}")
            self._cancel_order(self.active_limit_entry_order_details['orderId'])
            # self._reset_active_limit_entry_order_details() # Скинеться при отриманні CANCELED

        time.sleep(0.3) # Даємо біржі час на обробку скасувань

        close_order_side = "SELL" if self.current_position_on_exchange == "LONG" else "BUY"
        
        # Кількість для закриття вже має бути правильною (self.current_position_quantity)
        # і округленою відповідно до правил при відкритті/оновленні позиції.
        # Тут не потрібно повторно округлювати, якщо ми впевнені в self.current_position_quantity.
        # Але для безпеки можна передати через self._round_quantity_to_step_size
        quantity_to_close = self.current_position_quantity 
        
        close_order = self._place_market_order_to_close(
            order_closing_side=close_order_side, 
            quantity=quantity_to_close, 
            position_being_closed_side=self.current_position_on_exchange
        )
        
        if close_order and close_order.get('orderId'):
            self.logger.info(f"Market close order for {self.symbol} sent due to '{reason}'. OrderID: {close_order['orderId']}. Bot will be stopped.")
            # Стан позиції буде оновлено через WebSocket ORDER_TRADE_UPDATE / ACCOUNT_UPDATE
            # Тут ми не скидаємо self.current_position_on_exchange одразу, чекаємо підтвердження з WebSocket
        else:
            self.logger.error(f"FATAL for {self.symbol}: FAILED TO SEND MARKET CLOSE ORDER for '{reason}'. MANUAL INTERVENTION REQUIRED! Bot will be stopped.")
        
        if not self.bot_stopped_due_to_total_sl: # Зупиняємо бота, якщо він ще не був зупинений
            self.bot_stopped_due_to_total_sl = True
            self.logger.critical(f"Bot {self.symbol} stopped due to critical error: {reason}.")
            # Фінальне повідомлення буде надіслано в блоці finally методу run
    def _load_model_performance(self) -> dict:
            """Завантажує статистику (ПОВЕРНЕНО ДО СТАРОЇ ВЕРСІЇ)."""
            try:
                # Просто намагаємось прочитати
                with open(self.performance_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # Якщо файлу немає або він пошкоджений - повертаємо порожній словник
                return {}
            except Exception as e:
                self.logger.error(f"Не вдалося завантажити статистику моделей з {self.performance_file_path}: {e}", exc_info=True)
                return {} # Повертаємо порожній словник у разі будь-якої іншої помилки      
            
    def _save_model_performance(self, data: dict):
            """Зберігає статистику (ПОВЕРНЕНО ДО СТАРОЇ ВЕРСІЇ)."""
            try:
                # Режим 'w' створить файл або перезапише існуючий
                with open(self.performance_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                # self.logger.info(f"Статистику збережено у {self.performance_file_path}") # Можна додати, якщо хочеш бачити підтвердження
            except Exception as e:
                self.logger.error(f"CRITICAL SAVE ERROR: Не вдалося зберегти статистику моделей у файл {self.performance_file_path}: {e}", exc_info=True)

    def _format_performance_report(self, performance_data: dict, symbol: str) -> str:
            """
            ОНОВЛЕНА ВЕРСІЯ: Форматує звіт точно як на скріншоті,
            з рядком "Загальний" в кінці таблиці.
            """
            symbol_stats = performance_data.get(symbol, {})
            if not symbol_stats:
                return f"📊 **Звіт для {symbol}**: Статистика ще не зібрана."

            header = f"📊 **Звіт по точності прогнозів для {symbol}**\n\n"
            # Змінюємо заголовок, щоб він відповідав скріншоту
            table = "<pre>{:<20} | {:<10} | {:<5}\n".format("Модель", "Вінрейт", "%")
            table += "-"*42 + "\n"

            # Спочатку виводимо статистику по кожній окремій моделі
            # Сортуємо моделі за алфавітом для стабільного порядку
            model_keys = sorted([key for key in symbol_stats.keys() if key != 'Ensemble'])
            for model_name in model_keys:
                stats = symbol_stats.get(model_name, {'correct': 0, 'total': 0})
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                win_rate = (correct / total * 100) if total > 0 else 0
                
                table += "{:<20} | {:<10} | {:<5.1f}\n".format(model_name, f"{correct}/{total}", win_rate)
            
            table += "-"*42 + "\n"

            # В кінці додаємо загальний результат ансамблю
            ensemble_stats = symbol_stats.get('Ensemble', {'correct': 0, 'total': 0})
            ens_correct = ensemble_stats.get('correct', 0)
            ens_total = ensemble_stats.get('total', 0)
            ens_win_rate = (ens_correct / ens_total * 100) if ens_total > 0 else 0
            # Використовуємо "Загальний", як на скріншоті
            table += "{:<20} | {:<10} | {:<5.1f}\n".format("Загальний", f"{ens_correct}/{ens_total}", ens_win_rate)

            return header + table + "</pre>"

    def _send_telegram_report(self, message: str):
        """Надсилає фінальний звіт у Telegram, використовуючи HTML-розмітку."""
        if self.telegram_bot_token_for_logging and self.telegram_chat_id_for_logging_str:
            try:
                chat_id = int(self.telegram_chat_id_for_logging_str)
                url = f"https://api.telegram.org/bot{self.telegram_bot_token_for_logging}/sendMessage"
                payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
                requests.post(url, data=payload, timeout=10)
                self.logger.info("Фінальний звіт про ефективність моделей надіслано в Telegram.")
            except Exception as e:
                self.logger.error(f"Помилка надсилання звіту про ефективність в Telegram: {e}")

    def run(self):
        self.logger.info(f"Спроба запустити логіку бота для {self.symbol}")
        shutdown_reason = "несподіване завершення"
        self.bot_starting_virtual_capital = self.virtual_bot_capital_usdt
        self.current_position_on_exchange = None

        try:
            self._initialize_binance_client()
            self.logger.info(f"Перевірка існуючих відкритих позицій для {self.symbol} при старті...")
            existing_positions_list = self._get_open_positions(symbol_to_check=self.symbol)

            if existing_positions_list is None:
                self.logger.error(f"Не вдалося визначити існуючі позиції для {self.symbol}. Зупинка.")
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"Помилка API при перевірці позицій на старті для {self.symbol}"
            elif isinstance(existing_positions_list, list) and len(existing_positions_list) > 0:
                active_position_found = any(float(pos.get('positionAmt', 0)) != 0 for pos in existing_positions_list)
                if active_position_found:
                    pos_info_str = ", ".join([f"К-ть: {p.get('positionAmt',0)}, Вхід: {p.get('entryPrice',0)}" for p in existing_positions_list if float(p.get('positionAmt',0)) != 0])
                    self.logger.critical(f"!!! КРИТИЧНЕ ПОПЕРЕДЖЕННЯ ПРИ СТАРТІ для {self.symbol} !!!")
                    self.logger.critical(f"Знайдено існуючу позицію: {pos_info_str}.")
                    self.logger.critical("Бот розрахований на старт без існуючої позиції. Бот НЕ буде починати нові угоди.")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"знайдено активну позицію для {self.symbol} при старті"
            else: 
                self.logger.info(f"Існуючих відкритих позицій для {self.symbol} при старті не знайдено.")
            
            if self.should_stop_bot_flag_instance:
                self.logger.info(f"Бот для {self.symbol} зупинено через перевірки перед запуском: {shutdown_reason}")
                return

            if not self._get_symbol_trading_rules():
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"не вдалося отримати торгові правила для {self.symbol}"
                return
            
            if not self._get_and_store_leverage_data():
                self.logger.warning(f"Не вдалося завантажити дані про кредитне плече для {self.symbol}. Використовуються значення за замовчуванням.")
            
            if not self._set_isolated_margin():
                self.logger.warning(f"Не вдалося підтвердити ізольовану маржу для {self.symbol}.")

            local_initial_balance_float = self._get_account_balance_usdt()
            if local_initial_balance_float is None:
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"не вдалося отримати баланс акаунту для {self.symbol}"
                return
            self.initial_account_balance_usdt = Decimal(str(local_initial_balance_float))

            if self.total_account_stop_loss_usdt > 0:
                self.stop_bot_threshold_balance_usdt = self.initial_account_balance_usdt - self.total_account_stop_loss_usdt
                self.logger.info(f"Початковий баланс акаунту: {self.initial_account_balance_usdt:.2f} USDT. Глобальне значення SL: {self.total_account_stop_loss_usdt:.2f} USDT. Поріг зупинки бота (Акаунт): {self.stop_bot_threshold_balance_usdt:.2f} USDT для {self.symbol}.")
                if self.initial_account_balance_usdt < self.total_account_stop_loss_usdt:
                    self.logger.error(f"Початковий баланс ({self.initial_account_balance_usdt:.2f} USDT) менший за поріг TOTAL_ACCOUNT_STOP_LOSS_USDT ({self.total_account_stop_loss_usdt:.2f} USDT). Неможливо запустити {self.symbol}.")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"початковий баланс занадто низький відносно глобального SL"
            else:
                self.stop_bot_threshold_balance_usdt = Decimal('0') 
                self.logger.info(f"Початковий баланс акаунту: {self.initial_account_balance_usdt:.2f} USDT. TOTAL_ACCOUNT_STOP_LOSS_USDT не встановлено або нульове.")

            if self.should_stop_bot_flag_instance:
                self.logger.info(f"Бот для {self.symbol} зупинено на етапі ініціалізації: {shutdown_reason}")
                return

            # --- ЗМІНА: Завантажуємо ТІЛЬКИ основний ТФ ---
            self.klines_df_main = self._get_klines_df_rest(self.interval, self.limit_main_tf)
            # self.klines_df_higher_tf = ... (ВИДАЛЕНО)

            if self.klines_df_main.empty:
                self.logger.error(f"Не вдалося завантажити початкові історичні дані для {self.symbol}. Зупинка бота.")
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"не вдалося завантажити історичні дані для {self.symbol}"
            
            if self.should_stop_bot_flag_instance:
                self.logger.info(f"Бот для {self.symbol} зупинено: {shutdown_reason}")
                return

            self.last_processed_kline_open_time = self.klines_df_main.index[-1] if not self.klines_df_main.empty else None
            self.ws_client = UMFuturesWebsocketClient(
                on_message=self._on_websocket_message, 
                on_open=self._on_websocket_open,       
                on_close=self._on_websocket_close,     
                on_error=self._on_websocket_error      
            )
            self.listen_key = self._manage_user_data_stream()
            if not self.listen_key:
                self.logger.critical(f"Не вдалося отримати listen key для {self.symbol}. Зупинка.")
                self.should_stop_bot_flag_instance = True
                shutdown_reason = f"не вдалося отримати listen key для {self.symbol}"
            
            if self.should_stop_bot_flag_instance:
                self.logger.info(f"Бот для {self.symbol} зупинено перед підпискою на WS: {shutdown_reason}")
                return

            # --- ЗМІНА: Підписка ТІЛЬКИ на основний ТФ ---
            self.ws_client.user_data(listen_key=self.listen_key, id=0, callback=self._on_websocket_message)
            self.ws_client.kline(symbol=self.symbol, id=1, interval=self.interval, callback=self._on_websocket_message)
            # self.ws_client.kline(symbol=self.symbol, id=2, interval=self.selected_timeframe_higher... (ВИДАЛЕНО)

            self.logger.info(f"Вхід у головний операційний цикл для {self.symbol}.")
            websocket_restart_attempts = 0
            MAX_WEBSOCKET_RESTART_ATTEMPTS = self.config.get('MAX_WEBSOCKET_RESTART_ATTEMPTS', 3)

            while not self.should_stop_bot_flag_instance and not self.orchestrator_stop_event.is_set():
                current_time_main_loop = time.time()

                if os.path.exists(self.stop_flag_file_path_instance):
                    self.logger.critical(f"Файл-прапорець {self.stop_flag_file_path_instance} знайдено для {self.symbol}. Ініціація зупинки...")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"команда через файл-прапорець для {self.symbol}"
                    break 
                
                if self.orchestrator_stop_event.is_set():
                    self.logger.critical(f"Отримано сигнал зупинки від оркестратора для {self.symbol}. Ініціація зупинки...")
                    self.should_stop_bot_flag_instance = True
                    shutdown_reason = f"глобальний сигнал зупинки від оркестратора для {self.symbol}"
                    break
                
                if self.reconnect_websocket_flag:
                    if websocket_restart_attempts < MAX_WEBSOCKET_RESTART_ATTEMPTS:
                        websocket_restart_attempts += 1
                        self.logger.warning(f"RECONNECT_WEBSOCKET_FLAG is True for {self.symbol}. Attempting WS re-init #{websocket_restart_attempts}/{MAX_WEBSOCKET_RESTART_ATTEMPTS}...")
                        self.reconnect_websocket_flag = False

                        if self.ws_client and hasattr(self.ws_client, 'stop') and callable(self.ws_client.stop):
                            self.logger.info("Зупинка існуючого WebSocket клієнта перед реініціалізацією...")
                            try:
                                self.ws_client.stop()
                                time.sleep(3) 
                            except Exception as e_stop_ws:
                                self.logger.error(f"Помилка зупинки ws_client під час реініціалізації: {e_stop_ws}")
                        
                        self.logger.info("Реініціалізація WebSocket клієнта та підписок...")
                        try:
                            self.ws_client = UMFuturesWebsocketClient(
                                on_message=self._on_websocket_message, on_open=self._on_websocket_open,
                                on_close=self._on_websocket_close, on_error=self._on_websocket_error
                            )
                            temp_listen_key = self._manage_user_data_stream(current_listen_key=self.listen_key)          
                            if temp_listen_key:
                                if temp_listen_key != self.listen_key:
                                    self.logger.warning(f"Listen key для {self.symbol} змінився: {self.listen_key[:5] if self.listen_key else 'None'}... на {temp_listen_key[:5]}...")
                                self.listen_key = temp_listen_key
                                self.logger.info(f"Повторна підписка на User Data Stream для {self.symbol} з listenKey: {self.listen_key[:5]}...")
                                self.ws_client.user_data(listen_key=self.listen_key, id=0, callback=self._on_websocket_message)
                            else: 
                                self.logger.critical(f"Не вдалося отримати listen key під час WS реініціалізації для {self.symbol}. Зупинка.")
                                self.should_stop_bot_flag_instance = True
                                shutdown_reason = f"не вдалося отримати listen key при WS реініціалізації для {self.symbol}"
                            
                            if not self.should_stop_bot_flag_instance:
                                self.logger.info(f"Повторна підписка на KLINE стріми для {self.symbol}...")
                                # --- ЗМІНА: Тільки основний ТФ при реконнекті ---
                                self.ws_client.kline(symbol=self.symbol, id=1, interval=self.interval, callback=self._on_websocket_message)
                                # self.ws_client.kline(symbol=self.symbol, id=2... (ВИДАЛЕНО)
                                self.logger.info(f"WebSocket клієнт для {self.symbol} успішно реініціалізовано.")
                        except Exception as e_reconnect:
                            self.logger.error(f"ПОМИЛКА під час WS реініціалізації для {self.symbol}: {e_reconnect}. Спроба #{websocket_restart_attempts}.", exc_info=True)
                            if websocket_restart_attempts >= MAX_WEBSOCKET_RESTART_ATTEMPTS:
                                self.logger.critical(f"Всі {MAX_WEBSOCKET_RESTART_ATTEMPTS} спроби WS реініціалізації для {self.symbol} провалені. Зупинка.")
                                self.should_stop_bot_flag_instance = True
                                shutdown_reason = f"максимум спроб WS реініціалізації для {self.symbol}"
                        if self.should_stop_bot_flag_instance: break
                    else: 
                        self.logger.critical(f"Перевищено максимальну кількість спроб перезапуску WS ({MAX_WEBSOCKET_RESTART_ATTEMPTS}) для {self.symbol}. Зупинка.")
                        self.should_stop_bot_flag_instance = True
                        shutdown_reason = f"максимум спроб WS реініціалізації (прапорець не скинуто) для {self.symbol}"
                    if self.should_stop_bot_flag_instance: break

                if not self.should_stop_bot_flag_instance and not self.orchestrator_stop_event.is_set() and self.listen_key and \
                (self.last_listen_key_renewal_time == 0 or (current_time_main_loop - self.last_listen_key_renewal_time > 30 * 60)):
                    log_msg = "Початкова перевірка/оновлення listen key..." if self.last_listen_key_renewal_time == 0 else "Планова перевірка оновлення listen key..."
                    self.logger.info(f"{log_msg} для {self.symbol}")
                    new_key_candidate = self._manage_user_data_stream(current_listen_key=self.listen_key)
                    if new_key_candidate:
                        if new_key_candidate != self.listen_key:
                            self.logger.warning(f"Listen key для {self.symbol} змінився з {self.listen_key[:5] if self.listen_key else 'None'}... на {new_key_candidate[:5]}...!")
                            self.logger.info(f"Позначка для реініціалізації WebSocket для {self.symbol} через зміну listen key.")
                            self.listen_key = new_key_candidate
                            self.reconnect_websocket_flag = True 
                            websocket_restart_attempts = 0
                    else: 
                        self.logger.critical(f"Не вдалося оновити/отримати listen key для {self.symbol}. Зупинка.")
                        self.should_stop_bot_flag_instance = True
                        shutdown_reason = f"не вдалося оновити listen key для {self.symbol}"
                    if self.should_stop_bot_flag_instance: break

                if not self.should_stop_bot_flag_instance and not self.orchestrator_stop_event.is_set() and \
                current_time_main_loop - self.last_heartbeat_log_time >= (5*60):
                        uptime_seconds = current_time_main_loop - self.start_time
                        time_since_lk_renew_str = "Н/Д"
                        if self.last_listen_key_renewal_time > 0:
                            time_since_lk_renew_val = current_time_main_loop - self.last_listen_key_renewal_time
                            time_since_lk_renew_str = f"{time_since_lk_renew_val:.0f}с тому"
                        
                        self.logger.info(f"Heartbeat для {self.symbol}. Час роботи: {time.strftime('%H:%M:%S', time.gmtime(uptime_seconds))}. "
                                        f"Позиція: {self.current_position_on_exchange}. В.Капітал: {self.virtual_bot_capital_usdt:.4f} USDT. "
                                        f"ListenKey: {self.listen_key[:5] if self.listen_key else 'Н/Д'} (оновлено {time_since_lk_renew_str}). "
                                        f"ReconnectFlag: {self.reconnect_websocket_flag}, WS_Restarts: {websocket_restart_attempts}.")
                        self.last_heartbeat_log_time = current_time_main_loop

                time.sleep(1)
            
            if self.orchestrator_stop_event.is_set() and shutdown_reason == "несподіване завершення":
                shutdown_reason = f"глобальна зупинка оркестратора для {self.symbol}"
            elif self.should_stop_bot_flag_instance and shutdown_reason == "несподіване завершення":
                shutdown_reason = f"зупинка через індивідуальний прапорець для {self.symbol}"
            elif self.bot_stopped_due_to_total_sl and shutdown_reason == "несподіване завершення":
                shutdown_reason = f"внутрішній SL або критична помилка для {self.symbol}"
        
        except KeyboardInterrupt:
            self.logger.warning(f"KeyboardInterrupt отримано для {self.symbol}. Ініціація зупинки...")
            self.should_stop_bot_flag_instance = True
            shutdown_reason = f"KeyboardInterrupt для {self.symbol}"
        except Exception as e_fatal:
            self.logger.critical(f"Фатальна помилка в методі run для {self.symbol}: {e_fatal}", exc_info=True)
            self.should_stop_bot_flag_instance = True
            shutdown_reason = f"фатальна помилка в {self.symbol}: {str(e_fatal)[:200]}"
        finally:
            self.logger.info(f"Початок послідовності зупинки для {self.symbol} (Причина: {shutdown_reason})...")
            
            self._close_open_position_on_shutdown()
            self._ensure_all_orders_canceled_for_shutdown()

            if self.ws_client and hasattr(self.ws_client, 'stop') and callable(self.ws_client.stop):
                self.logger.info(f"Спроба закрити WebSocket з'єднання для {self.symbol} у блоці finally...")
                try:
                    self.ws_client.stop() 
                    self.logger.info(f"Команду зупинки WebSocket клієнта для {self.symbol} надіслано успішно.")
                except Exception as e_stop: 
                    self.logger.error(f"Помилка під час ws_client.stop() для {self.symbol} у блоці finally: {e_stop}", exc_info=True)
            else:
                self.logger.info(f"WebSocket клієнт для {self.symbol} не був ініціалізований або вже очищений.")
            
            if os.path.exists(self.stop_flag_file_path_instance):
                try:
                    os.remove(self.stop_flag_file_path_instance)
                    self.logger.info(f"Індивідуальний файл-прапорець {self.stop_flag_file_path_instance} видалено при завершенні роботи {self.symbol}.")
                except OSError as e_rem:
                    self.logger.error(f"Помилка видалення індивідуального файлу-прапорця {self.stop_flag_file_path_instance} для {self.symbol}: {e_rem}")

            final_bot_status_message = f"🤖 Торговий бот для *{self.symbol}* завершив роботу.\n🏁 *Причина*: `{shutdown_reason}`."
            self._send_final_telegram_message(final_bot_status_message)
            
            self.logger.info(f"Торговий бот для {self.symbol} ({threading.current_thread().name}) завершив свій цикл run.") 
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] Торговий бот ({self.symbol} - {threading.current_thread().name}) завершив роботу.")

def prepare_for_backtest(self):
    print("Підготовка екземпляру бота для бек-тестування...")
    try:
        # Ініціалізуємо клієнт та завантажуємо правила торгівлі
        self.client = UMFutures()
        
        # Просто викликаємо метод. Він заповнить self.symbol_trading_rules всередині себе.
        # А ми перевіримо, чи повернув він True (успіх).
        if not self._get_symbol_trading_rules():
            # Якщо метод повернув False, то сталася помилка
            raise ValueError("Не вдалося завантажити правила торгівлі під час ініціалізації для бек-тесту.")
        
        # У цьому місці self.symbol_trading_rules вже буде правильним словником.
        # Решта коду залишається без змін.
        
        self.strategy_params = self.config.get('strategy_params', {})
        self.n1 = self.strategy_params.get('n1', 20)
        self.n2 = self.strategy_params.get('n2', 10)
        self.atr_multiplier = self.strategy_params.get('atr_multiplier', 2.0)
        self.entry_retracement_pct = Decimal(str(self.strategy_params.get('entry_retracement_pct', '0.5')))
        self.tp_multiplier = Decimal(str(self.strategy_params.get('TP_MULTIPLIER', '1.0')))
        self.sl_multiplier = Decimal(str(self.strategy_params.get('SL_MULTIPLIER', '0.2')))

        print("Ініціалізація для бек-тесту успішна.")
    except Exception as e:
        print(f"Помилка під час спеціальної ініціалізації для бек-тесту: {e}")
        raise

    
    def setup_for_backtest(self):
        """Готує екземпляр бота для бек-тестування."""
        print("Налаштування екземпляру бота для бек-тесту...")
        # Явно встановлюємо параметри з конфігурації
        strategy_params = self.config.get('strategy_params', {})
        self.n1 = strategy_params.get('n1', 20)
        self.n2 = strategy_params.get('n2', 10)
        self.atr_multiplier = strategy_params.get('atr_multiplier', 2.0)
        self.entry_retracement_pct = Decimal(str(strategy_params.get('entry_retracement_pct', '0.5')))
        self.tp_multiplier = Decimal(str(strategy_params.get('TP_MULTIPLIER', '1.0')))
        self.sl_multiplier = Decimal(str(strategy_params.get('SL_MULTIPLIER', '0.2')))
        print(f"Параметри для бектесту встановлено: n1={self.n1}, SL_Multiplier={self.sl_multiplier}")

    

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
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN") # Для TelegramLoggingHandler екземплярів
        self.telegram_chat_id_str = os.getenv("YOUR_TELEGRAM_CHAT_ID") # Для TelegramLoggingHandler екземплярів

        # Налаштування логгера Оркестратора
        self.logger = logging.getLogger("Orchestrator")
        if not self.logger.handlers: # Уникаємо дублювання, якщо скрипт запускається декілька разів в одному середовищі
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d UTC %(levelname)-8s %(name)-15s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # Можна додати файловий логгер для оркестратора
            # log_file_orchestrator = os.path.join(SCRIPT_DIR_NEW_LOGIC, "orchestrator_main.log")
            # fh = logging.FileHandler(log_file_orchestrator, mode='a', encoding='utf-8')
            # fh.setFormatter(formatter)
            # self.logger.addHandler(fh)

        if not all([self.api_key, self.api_secret]):
            self.logger.critical("Binance API_KEY або API_SECRET не знайдено. Оркестратор не може стартувати.")
            raise ValueError("Binance API keys not found for Orchestrator.")
        
        self.logger.info("Оркестратор ініціалізовано.")

    def _check_if_models_exist(self, symbol: str) -> bool:
        """
        Перевіряє наявність тільки Transformer моделі та скейлера.
        """
        scaler_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{symbol}_StepPredictor_Scaler_3D.joblib")
        model_path = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"{symbol}_StepPredictor_Transformer_model.keras")

        if not os.path.exists(scaler_path):
            self.logger.warning(f"3D Скейлер для {symbol} не знайдено ({scaler_path}).")
            return False
        if not os.path.exists(model_path):
            self.logger.warning(f"Transformer модель для {symbol} не знайдено ({model_path}).")
            return False
        
        return True
    
    def _load_raw_config_from_file(self) -> list:
        self.logger.debug(f"Спроба завантажити конфігурацію з {self.config_file_path}")
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                loaded_configs = json.load(f)
            if not isinstance(loaded_configs, list):
                self.logger.error(f"Конфігурація у файлі {self.config_file_path} не є списком. Повертаю порожній список.")
                return []
            self.logger.info(f"Успішно завантажено {len(loaded_configs)} 'сирих' конфігурацій з {self.config_file_path}.")
            return loaded_configs
        except FileNotFoundError:
            self.logger.warning(f"Файл конфігурації {self.config_file_path} не знайдено. Повертаю порожній список.")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Помилка декодування JSON з файлу {self.config_file_path}: {e}. Повертаю порожній список.")
            return []
        except Exception as e:
            self.logger.error(f"Не вдалося завантажити конфігурації з {self.config_file_path}: {e}", exc_info=True)
            return []

    def _prepare_bots_configurations(self, raw_configs: list) -> list:
        if not raw_configs:
            return []

        # Розраховуємо глобальний SL на основі капіталу з файлу
        total_initial_capital_sum = sum(
            Decimal(str(item.get("config", {}).get('INITIAL_CAPITAL_PER_TRADE_USDT', '0.0')))
            for item in raw_configs
        )
        global_total_account_sl_value = (total_initial_capital_sum / Decimal('2.0')) if total_initial_capital_sum > 0 else Decimal('0.0')

        processed_configurations = []
        for bot_conf_item in raw_configs:
            symbol = bot_conf_item["symbol"]
            config_from_file = bot_conf_item.get("config", {})
            
            # Якщо бот вже працює, ми беремо його поточний капітал, а не початковий
            if symbol in self.active_bot_instances:
                current_instance_capital = self.active_bot_instances[symbol].virtual_bot_capital_usdt
                config_from_file['VIRTUAL_BOT_CAPITAL_USDT'] = str(current_instance_capital)
                self.logger.info(f"Збереження існуючого капіталу для активного бота {symbol}: {current_instance_capital:.4f} USDT")

            # Додаємо розрахований глобальний SL
            config_from_file['TOTAL_ACCOUNT_STOP_LOSS_USDT'] = str(global_total_account_sl_value)

            processed_configurations.append({
                "symbol": symbol,
                "interval": config_from_file.get("interval", "5m"),
                "config": config_from_file
            })
        self.logger.info(f"Підготовлено {len(processed_configurations)} повних конфігурацій ботів.")
        return processed_configurations
    # --- КІНЕЦЬ ВИПРАВЛЕНОГО БЛОКУ 2 ---

    def _start_bot_instance(self, bot_params: dict):
        symbol = bot_params["symbol"]
        if symbol in self.active_bot_instances:
            self.logger.warning(f"Бот для {symbol} вже запущений. Пропуск запуску.")
            return

        self.logger.info(f"Спроба запуску бота для {symbol} з інтервалом {bot_params.get('interval', 'N/A')}")
        
        individual_stop_flag = os.path.join(SCRIPT_DIR_NEW_LOGIC, f"STOP_TRADING_BOT_{symbol}.flag")
        if os.path.exists(individual_stop_flag):
            try:
                os.remove(individual_stop_flag)
                self.logger.info(f"Видалено старий індивідуальний прапорець зупинки для {symbol}: {individual_stop_flag}")
            except OSError as e:
                self.logger.error(f"Не вдалося видалити старий індивідуальний прапорець зупинки для {symbol}: {e}")

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
            self.logger.error(f"Помилка ініціалізації TradingBotInstance для {symbol}: {e_init}", exc_info=True)
            return # Не запускаємо потік, якщо ініціалізація не вдалася
        
        # --- ПОЧАТОК ДІАГНОСТИЧНОГО БЛОКУ ---
        print("\n" + "="*20 + f" ДІАГНОСТИКА ОБ'ЄКТА ДЛЯ {symbol} " + "="*20)
        try:
            print(f"ТИП ОБ'ЄКТА: {type(instance)}")
            print(f"Чи є у об'єкта атрибут 'run'? -> {hasattr(instance, 'run')}")
            print("СПИСОК ДОСТУПНИХ АТРИБУТІВ І МЕТОДІВ:")
            # Виводимо список методів, які не починаються з '__'
            print([attr for attr in dir(instance) if not attr.startswith('__')])
        except Exception as e_diag:
            print(f"Помилка під час діагностики: {e_diag}")
        print("="*60 + "\n")
        # --- КІНЕЦЬ ДІАГНОСТИЧНОГО БЛОКУ ---

          
        thread = threading.Thread(target=instance.run, name=f"BotThread-{symbol}")
        self.active_bot_instances[symbol] = instance
        self.active_bot_threads[symbol] = thread
        thread.start()
        self.logger.info(f"Бот для {symbol} успішно запущено в потоці {thread.name}.")

    def _stop_bot_instance(self, symbol: str, reason: str = "config_update"):
        if symbol not in self.active_bot_instances:
            self.logger.debug(f"Бот для {symbol} не знайдено для зупинки (можливо, вже зупинено).")
            return

        self.logger.info(f"Ініціація зупинки бота для {symbol}. Причина: {reason}.")
        instance = self.active_bot_instances.get(symbol)
        thread = self.active_bot_threads.get(symbol)

        if instance:
            instance.should_stop_bot_flag_instance = True
        
        if thread:
            self.logger.info(f"Очікування завершення потоку {thread.name} для {symbol} (таймаут 60с)...")
            thread.join(timeout=60)
            if thread.is_alive():
                self.logger.warning(f"Потік {thread.name} для {symbol} не завершився коректно після таймауту.")
            else:
                self.logger.info(f"Потік {thread.name} для {symbol} успішно завершено.")
        
        # Видаляємо зі списків активних незалежно від того, чи завершився потік вчасно
        if symbol in self.active_bot_instances: del self.active_bot_instances[symbol]
        if symbol in self.active_bot_threads: del self.active_bot_threads[symbol]
        self.logger.info(f"Екземпляр бота для {symbol} видалено зі списків активних в оркестраторі.")


    def _load_and_reconcile_bots(self):
        self.logger.info(f"Перевірка змін та узгодження ботів згідно {self.config_file_path}...")
        
        full_new_bot_configurations = self._load_and_prepare_configs()

        new_config_symbols = set(bot_conf["symbol"] for bot_conf in full_new_bot_configurations)
        current_running_symbols = set(self.active_bot_instances.keys())

        # --- КРОК 1: Зупиняємо ботів, яких видалили з конфігурації ---
        symbols_to_stop = current_running_symbols - new_config_symbols
        if symbols_to_stop:
            self.logger.info(f"Символи для зупинки (видалені з конфігу): {symbols_to_stop}")
            for symbol in symbols_to_stop:
                self._stop_bot_instance(symbol, reason="Видалено з конфігурації")

        # --- КРОК 2: Запускаємо тільки нових ботів ---
        symbols_to_start = new_config_symbols - current_running_symbols
        if symbols_to_start:
            self.logger.info(f"Символи для запуску (нові в конфігу): {symbols_to_start}")
            for symbol_to_add in symbols_to_start:
                # Знаходимо повну конфігурацію для нового бота
                bot_params = next((conf for conf in full_new_bot_configurations if conf["symbol"] == symbol_to_add), None)
                if bot_params:
                    # Перевірка наявності моделей для нового бота
                    if bot_params.get("config", {}).get("USE_ML_FILTER", False):
                        if not self._check_if_models_exist(symbol_to_add):
                            self.logger.warning(f"Пропуск запуску нового бота {symbol_to_add}, оскільки моделі не навчені.")
                            continue
                    
                    time.sleep(2) # Невелика затримка між запусками
                    self._start_bot_instance(bot_params)
                else:
                    self.logger.error(f"Не вдалося знайти конфігурацію для нового символу {symbol_to_add} для запуску.")

        # --- КРОК 3: Ігноруємо ботів, що вже працюють і не змінилися ---
        symbols_to_keep = current_running_symbols.intersection(new_config_symbols)
        if symbols_to_keep:
            self.logger.info(f"Символи, що продовжують роботу без змін: {symbols_to_keep}")

        # Оновлюємо час останньої модифікації файлу
        if os.path.exists(self.config_file_path):
            self.last_config_mtime = os.path.getmtime(self.config_file_path)
        else:
            self.last_config_mtime = 0

        self.logger.info("Узгодження ботів завершено.")

    def _load_and_prepare_configs(self):
        """Допоміжна функція для завантаження та підготовки конфігурацій."""
        raw_configs = self._load_raw_config_from_file()
        return self._prepare_bots_configurations(raw_configs)

    def run_main_loop(self):
        self.logger.info("Оркестратор запущено. Початок основного циклу моніторингу...")
        
        # Початкове завантаження та запуск ботів
        if os.path.exists(self.config_file_path):
            try:
                self.last_config_mtime = os.path.getmtime(self.config_file_path)
            except OSError:
                self.logger.error(f"Не вдалося отримати час модифікації для {self.config_file_path} при старті.")
                self.last_config_mtime = 0
            self._load_and_reconcile_bots()
        else:
            self.logger.warning(f"Початковий файл конфігурації {self.config_file_path} не знайдено. Оркестратор очікуватиме на його створення або прапорець RELOAD.")
            self.last_config_mtime = 0

        try:
            while not self.global_stop_event.is_set():
                if os.path.exists(GLOBAL_STOP_FLAG_FILE_PATH):
                    self.logger.critical(f"!!! {GLOBAL_STOP_FLAG_FILE_PATH} виявлено! Ініціація зупинки оркестратора та всіх ботів.")
                    self.global_stop_event.set()
                    try:
                        os.remove(GLOBAL_STOP_FLAG_FILE_PATH)
                        self.logger.info(f"Файл {GLOBAL_STOP_FLAG_FILE_PATH} видалено оркестратором.")
                    except OSError as e:
                        self.logger.error(f"Помилка видалення {GLOBAL_STOP_FLAG_FILE_PATH} оркестратором: {e}")
                    break 

                if os.path.exists(RELOAD_CONFIG_FLAG_FILE_PATH):
                    self.logger.info(f"Прапорець {RELOAD_CONFIG_FLAG_FILE_PATH} виявлено. Перезавантаження конфігурації.")
                    try:
                        os.remove(RELOAD_CONFIG_FLAG_FILE_PATH)
                        self.logger.info(f"Файл {RELOAD_CONFIG_FLAG_FILE_PATH} видалено.")
                    except OSError as e:
                        self.logger.error(f"Помилка видалення {RELOAD_CONFIG_FLAG_FILE_PATH}: {e}")
                    
                    self._load_and_reconcile_bots() # Запускаємо узгодження
                else: # Якщо прапорця немає, можна перевіряти час модифікації файлу (як альтернатива або додаток)
                    if os.path.exists(self.config_file_path):
                        try:
                            current_mtime = os.path.getmtime(self.config_file_path)
                            if current_mtime != self.last_config_mtime:
                                self.logger.info(f"Виявлено зміну файлу конфігурації {self.config_file_path} за часом модифікації.")
                                self._load_and_reconcile_bots()
                            # self.last_config_mtime оновлюється всередині _load_and_reconcile_bots
                        except OSError:
                            self.logger.debug(f"Не вдалося перевірити час модифікації {self.config_file_path}.")
                    elif self.last_config_mtime != 0 : # Файл був, але тепер його немає
                        self.logger.warning(f"Файл конфігурації {self.config_file_path} був видалений. Зупинка всіх ботів.")
                        self._load_and_reconcile_bots() # Це викличе зупинку всіх, бо конфіг буде порожнім


                # Перевірка стану потоків
                for symbol, thread in list(self.active_bot_threads.items()):
                    if not thread.is_alive():
                        self.logger.warning(f"Потік для {symbol} ({thread.name}) несподівано завершився.")
                        if symbol in self.active_bot_instances: del self.active_bot_instances[symbol]
                        del self.active_bot_threads[symbol]
                        # Можна додати логіку автоматичного перезапуску тут, якщо потрібно

                time.sleep(10) # Інтервал перевірки прапорців та стану

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt отримано оркестратором. Ініціація зупинки.")
            self.global_stop_event.set()
        except Exception as e_main_loop:
            self.logger.critical(f"Неперехоплена помилка в головному циклі оркестратора: {e_main_loop}", exc_info=True)
            self.global_stop_event.set() # Намагаємося зупинитися коректно
        finally:
            self.logger.info("Початок процедури фінальної зупинки оркестратора...")
            if not self.global_stop_event.is_set(): # Якщо подія ще не встановлена (наприклад, не було KeyboardInterrupt)
                self.logger.info("Встановлення global_stop_event у finally блоці.")
                self.global_stop_event.set()

            active_symbols_at_shutdown = list(self.active_bot_instances.keys())
            if active_symbols_at_shutdown:
                self.logger.info(f"Зупинка решти активних ботів при завершенні оркестратора: {active_symbols_at_shutdown}")
                for symbol in active_symbols_at_shutdown:
                    self._stop_bot_instance(symbol, reason="orchestrator_global_shutdown")
            
            # Додаткове очікування для потоків, якщо вони ще не завершились
            final_threads_to_join = [t for t in self.active_bot_threads.values() if t.is_alive()]
            if final_threads_to_join:
                 self.logger.info(f"Очікування завершення {len(final_threads_to_join)} потоків при фінальній зупинці...")
                 for thread in final_threads_to_join:
                     thread.join(timeout=15) # Збільшений таймаут для фінального join
                     if thread.is_alive():
                         self.logger.warning(f"Потік {thread.name} все ще активний після фінального join.")
            
            # Спроба видалити глобальний прапорець, якщо він ще існує
            if os.path.exists(GLOBAL_STOP_FLAG_FILE_PATH):
                try:
                    os.remove(GLOBAL_STOP_FLAG_FILE_PATH)
                    self.logger.info(f"Файл {GLOBAL_STOP_FLAG_FILE_PATH} остаточно видалено оркестратором при завершенні.")
                except OSError: pass
            if os.path.exists(RELOAD_CONFIG_FLAG_FILE_PATH): # Також чистимо цей
                try:
                    os.remove(RELOAD_CONFIG_FLAG_FILE_PATH)
                except OSError: pass
            
            self.logger.info("Оркестратор завершив свою роботу остаточно.")

# --- Головний блок виконання ---
if __name__ == '__main__':
    # Вирішуємо проблему з кодуванням у Windows консолі
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except TypeError:
            pass
            
    parser = argparse.ArgumentParser(description="Універсальний скрипт для торгівлі та навчання моделей.")
    
    # --- РЕЖИМИ ЗАПУСКУ ---
    group = parser.add_mutually_exclusive_group(required=True) 
    group.add_argument("--train-and-validate", action='store_true', help="Запустити старий монолітний цикл (Навчання 80% + Валідація 20%).")
    group.add_argument("--train-only", action='store_true', help="ТІЛЬКИ навчити 'золоту' модель на вказаному діапазоні і зберегти її.")
    group.add_argument("--validate-only", action='store_true', help="ТІЛЬКИ завантажити 'золоту' модель і провалідувати її на вказаному діапазоні.")
    group.add_argument("--run-single", type=str, metavar="SYMBOL", help="Запустити у режимі ОДИНОЧНОГО БОТА (реальна торгівля).")
    
    # --- ПОЧАТОК ВИПРАВЛЕННЯ: Додано новий аргумент --run-orchestrator ---
    group.add_argument("--run-orchestrator", action='store_true', help="Запустити Оркестратор (для виклику з telegram_manager).")
    # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---
    
    # --- Загальні аргументи ---
    parser.add_argument("--symbol", type=str, help="Торговий символ (обов'язковий для всіх режимів, крім оркестратора)")
    parser.add_argument("--config", type=str, default=DEFAULT_BOT_CONFIG_FILE, help="Шлях до файлу конфігурації (використовується --run-orchestrator).")
    parser.add_argument("--no-progress-bar", action='store_true', help="Вимкнути відображення прогрес-барів tqdm.")
    
    # --- ПАРАМЕТРИ ДЛЯ НАВЧАННЯ/ВАЛІДАЦІЇ ---
    train_group = parser.add_argument_group('Параметри для навчання/валідації')
    train_group.add_argument("--lookback-months", type=float, default=12.0, help="[Для --train-and-validate]: Завантажити дані за останні N місяців.") 
    train_group.add_argument("--train-start", type=str, help="[Для --train-only]: Дата початку навчання (напр. '2025-01-01 00:00:00')")
    train_group.add_argument("--train-end", type=str, help="[Для --train-only]: Дата кінця навчання (напр. '2025-10-19 00:00:00')")
    train_group.add_argument("--test-start", type=str, help="[Для --validate-only]: Дата початку валідації (напр. '2025-10-19 00:00:00')")
    train_group.add_argument("--test-end", type=str, help="[Для --validate-only]: Дата кінця валідації (напр. '2025-10-21 00:00:00')")

    
    args = parser.parse_args()

    # --- ВАЛІДАЦІЯ АРГУМЕНТІВ ---
    if args.run_single:
        args.symbol = args.run_single.upper() 
        
    if (args.train_and_validate or args.train_only or args.validate_only) and not args.symbol:
         parser.error("Аргумент --symbol є обов'язковим для --train-and-validate, --train-only, --validate-only")

    if args.train_only and (not args.train_start or not args.train_end):
        parser.error("--train-start та --train-end є обов'язковими для --train-only")
        
    if args.validate_only and (not args.test_start or not args.test_end):
        parser.error("--test-start та --test-end є обов'язковими для --validate-only")
        
    if args.run_orchestrator and not args.config:
        parser.error("--config є обов'язковим для --run-orchestrator")


    # --- ВИКЛИК ПОТРІБНОЇ ФУНКЦІЇ ---
    if args.train_and_validate:
        main_logger.info("Запуск у режимі [Train-and-Validate (80/20)]...")
        run_training_process(args) 
        
    elif args.train_only:
        main_logger.info("Запуск у режимі [Train-Only]...")
        run_training_only(args)
        
    elif args.validate_only:
        main_logger.info("Запуск у режимі [Validate-Only]...")
        run_validation_only(args)

    elif args.run_single:
        # --- ПОЧАТОК ВИПРАВЛЕННЯ: Завантажуємо ключі для торгівлі ---
        main_logger.info("Завантаження .env ключів для режиму [run-single]...")
        load_dotenv(dotenv_path="key.env")
        api_key_main = os.getenv('BINANCE_API_KEY')
        api_secret_main = os.getenv('BINANCE_API_SECRET')
        telegram_token_main = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id_main = os.getenv('YOUR_TELEGRAM_CHAT_ID')

        if not api_key_main or not api_secret_main:
            main_logger.critical("Ключі API Binance не знайдено у key.env. Робота неможлива.")
            sys.exit(1)
        # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---

        symbol_to_run = args.run_single.upper()
        main_logger.info(f"--- РЕЖИМ ОДИНОЧНОГО ЗАПУСКУ для {symbol_to_run} ---")
        
        with open(args.config, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        
        bot_config_data = next((item for item in full_config if item["symbol"] == symbol_to_run), None)
        
        if not bot_config_data:
            main_logger.error(f"Не знайдено конфігурацію для символу {symbol_to_run} у файлі {args.config}")
            sys.exit(1)

        bot_instance = TradingBotInstance(
            symbol=bot_config_data["symbol"],
            interval=bot_config_data.get("interval", "5m"),
            api_key=api_key_main,
            api_secret=api_secret_main,
            bot_config=bot_config_data.get("config", {}),
            telegram_bot_token=telegram_token_main,
            telegram_chat_id_str=telegram_chat_id_main
        )
        bot_instance.run()
    
    # --- ПОЧАТОК ВИПРАВЛЕННЯ: Окрема логіка для Оркестратора ---
    elif args.run_orchestrator:
        # Завантажуємо ключі для Оркестратора
        main_logger.info("Завантаження .env ключів для режиму [run-orchestrator]...")
        load_dotenv(dotenv_path="key.env")
        
        # Перевіряємо, чи ключі завантажились *до* ініціалізації Оркестратора
        if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
            main_logger.critical("Ключі API Binance не знайдено у key.env. Запуск Оркестратора неможливий.")
            sys.exit(1)
            
        main_logger.info(f"Запуск оркестратора з файлом конфігурації: {args.config}")
        orchestrator = Orchestrator(config_file_path=args.config)
        orchestrator.run_main_loop()
    # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---
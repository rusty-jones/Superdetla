import streamlit as st
import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import os

# --- Constants ---
WATCHLIST_FILE = "market_watches.json"

# --- Persistence Functions ---
def load_market_watches():
    """Loads market watch lists from the JSON file."""
    if not os.path.exists(WATCHLIST_FILE):
        return {"Market Watch 1": [], "Market Watch 2": [], "Market Watch 3": []}
    try:
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"Market Watch 1": [], "Market Watch 2": [], "Market Watch 3": []}

def save_market_watches(watches):
    """Saves the market watch lists to the JSON file."""
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watches, f, indent=4)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)

# Ticker mapping
ticker_mapping = {
    "nifty50": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN"
}

# Timeframe and period mapping
timeframes = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1wk", "1mo": "1mo", "3mo": "3mo"
}
periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

# Initialize session state
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
if 'zones_list' not in st.session_state:
    st.session_state.zones_list = {}
if 'trades_list' not in st.session_state:
    st.session_state.trades_list = {}
if 'metrics_list' not in st.session_state:
    st.session_state.metrics_list = {}
if 'equity_list' not in st.session_state:
    st.session_state.equity_list = {}
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'aligned_zones' not in st.session_state:
    st.session_state.aligned_zones = {}
if 'fresh_zones' not in st.session_state:
    st.session_state.fresh_zones = {}
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = {}
if 'zone_groups_debug' not in st.session_state:
    st.session_state.zone_groups_debug = {}
if 'all_zones_debug' not in st.session_state:
    st.session_state.all_zones_debug = {}
if 'proposed_trade' not in st.session_state:
    st.session_state.proposed_trade = {}
if 'plot_data_ready' not in st.session_state:
    st.session_state.plot_data_ready = False
if 'backtest_data_ready' not in st.session_state:
    st.session_state.backtest_data_ready = False
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = []

# Function to fetch and clean data
def fetch_and_process_data(tickers, periods_intervals):
    all_data = {}
    for ticker in tickers:
        all_data[ticker] = {}
        for period, intervals in periods_intervals.items():
            for interval in intervals:
                try:
                    data = yf.download(ticker, period=period, interval=interval)
                    st.session_state.trade_log.append(f"Fetched data for {ticker} ({period}, {interval}): Length={len(data)}")
                    if data.empty:
                        st.session_state.trade_log.append(f"No data fetched for {ticker} ({period}, {interval})")
                        continue
                    data.reset_index(inplace=True)
                    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                    data.set_index('Date', inplace=True)
                    if len(data) >= 10000:
                        st.session_state.trade_log.append(f"Data exceeds 10000 candles for {ticker} ({period}, {interval}), truncated")
                        data = data.iloc[-10000:]
                    all_data[ticker][(period, interval)] = data
                except Exception as e:
                    st.session_state.trade_log.append(f"Error fetching {ticker} ({period}, {interval}): {str(e)}")
    return all_data

# Zone identification
def identify_zones(data):
    zones = []
    if len(data) < 2:
        st.session_state.trade_log.append("Insufficient data for zone identification")
        return zones
    for i in range(1, len(data) - 1):
        body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
        total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
        if body_size_base > 0.5 * total_height_base or total_height_base == 0:
            continue
        body_size_follow = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_follow = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        if body_size_follow < 0.75 * total_height_follow or total_height_follow == 0:
            continue
        date = data.index[i].to_pydatetime()
        ist = pytz.timezone('Asia/Kolkata')
        index_tz = data.index.tz
        if index_tz:
            zone_date = date.replace(tzinfo=index_tz) if date.tzinfo is None else date.astimezone(index_tz)
        else:
            zone_date = ist.localize(date) if date.tzinfo is None else date.astimezone(ist)
        if data['close'].iloc[i+1] > data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'demand', 'level': data['low'].iloc[i]})
        elif data['close'].iloc[i+1] < data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'supply', 'level': data['high'].iloc[i]})
    st.session_state.trade_log.append(f"Identified {len(zones)} zones for data length {len(data)}")
    return zones

# Approach detection and labeling
def find_approaches_and_labels(data, zones):
    instances = []
    if not zones or len(data) < 2:
        st.session_state.trade_log.append("No zones or insufficient data for approach detection")
        return instances
    for zone in zones:
        zone_date = zone['date']
        zone_level = zone['level']
        zone_type = zone['type']
        index_tz = data.index.tz
        if index_tz:
            zone_date = zone_date.astimezone(index_tz)
        future_data = data[data.index > zone_date]
        if zone_type == 'demand':
            approaches = future_data[(future_data['low'] >= zone_level * 0.99) & (future_data['low'] <= zone_level * 1.01)]
        else:
            approaches = future_data[(future_data['high'] >= zone_level * 0.99) & (future_data['high'] <= zone_level * 1.01)]
        for approach_date in approaches.index:
            approach_price = data.loc[approach_date, 'close']
            post_approach = data[data.index > approach_date]
            if zone_type == 'demand':
                break_level = zone_level * 0.995
                target_level = approach_price * 1.02
                hit_break = (post_approach['low'] <= break_level).any()
                hit_target = (post_approach['high'] >= target_level).any()
            else:
                break_level = zone_level * 1.005
                target_level = approach_price * 0.98
                hit_break = (post_approach['high'] >= break_level).any()
                hit_target = (post_approach['low'] <= target_level).any()
            if hit_break and hit_target:
                break_idx = post_approach[post_approach['low'] <= break_level].index[0] if zone_type == 'demand' else post_approach[post_approach['high'] >= break_level].index[0]
                target_idx = post_approach[post_approach['high'] >= target_level].index[0] if zone_type == 'demand' else post_approach[post_approach['low'] <= target_level].index[0]
                outcome = 'held' if target_idx < break_idx else 'broke'
            elif hit_target:
                outcome = 'held'
            elif hit_break:
                outcome = 'broke'
            else:
                continue
            prev_approaches = len(data[(data.index > zone_date) & (data.index < approach_date) & 
                                      (data['low' if zone_type == 'demand' else 'high'] >= zone_level * 0.99) & 
                                      (data['low' if zone_type == 'demand' else 'high'] <= zone_level * 1.01)])
            features = {'prev_approaches': prev_approaches}
            instances.append({'features': features, 'outcome': outcome})
    st.session_state.trade_log.append(f"Found {len(instances)} approach instances for {len(zones)} zones")
    return instances

# Calculate zone significance
def calculate_zone_significance(data, zone, timeframe_idx):
    approaches = len(find_approaches_and_labels(data, [zone]))
    ist = pytz.timezone('Asia/Kolkata')
    age = (datetime.now(ist) - zone['date'].astimezone(ist)).total_seconds() / (3600 * 24)
    timeframe_weight = {0: 2.0, 1: 1.5, 2: 1.2, 3: 1.0}.get(timeframe_idx, 1.0)
    score = (approaches * 10 + age * 0.1) * timeframe_weight
    st.session_state.trade_log.append(f"Zone at {zone['level']:.2f} ({zone['type']}, TF {timeframe_idx+1}): approaches={approaches}, age={age:.1f} days, score={score:.2f}")
    return score, approaches, age

# Build relationship chart
def build_relationship_chart(dfs, zones_list, timeframes_list, ticker, tolerance):
    relationship_chart = []
    all_zones = []
    st.session_state.fresh_zones[ticker] = []
    st.session_state.zone_groups_debug[ticker] = []
    st.session_state.all_zones_debug[ticker] = []
    
    for idx, (df, zones, tf) in enumerate(zip(dfs[ticker], zones_list[ticker], timeframes_list)):
        if zones and df is not None and not df.empty:
            for zone in zones:
                try:
                    score, approaches, age = calculate_zone_significance(df, zone, idx)
                    zone_info = {
                        'level': zone['level'],
                        'type': zone['type'],
                        'timeframe': tf,
                        'score': score,
                        'approaches': approaches,
                        'age': age,
                        'date': zone['date'],
                        'tf_idx': idx
                    }
                    all_zones.append(zone_info)
                    st.session_state.all_zones_debug[ticker].append({
                        'level': zone['level'],
                        'type': zone['type'],
                        'timeframe': tf,
                        'approaches': approaches,
                        'age': age
                    })
                    if approaches == 0:
                        st.session_state.fresh_zones[ticker].append({
                            'level': zone['level'],
                            'type': zone['type'],
                            'timeframe': tf,
                            'age': age,
                            'date': zone['date']
                        })
                        st.session_state.trade_log.append(f"Added fresh zone at {zone['level']:.2f} ({zone['type']}) in {tf} for {ticker}")
                except Exception as e:
                    st.session_state.trade_log.append(f"Error calculating significance for {ticker} (TF {tf}): {str(e)}")
                time.sleep(0.01) # Small delay to prevent CPU maxing out
    
    if not all_zones:
        st.session_state.trade_log.append(f"No valid zones processed for {ticker}")
        return relationship_chart
    
    all_zones.sort(key=lambda x: (x['level'], x['type']))
    current_group = []
    for zone in all_zones:
        if not current_group or (zone['type'] == current_group[0]['type'] and abs(zone['level'] - current_group[0]['level']) / current_group[0]['level'] <= tolerance / 100):
            current_group.append(zone)
        else:
            if len(current_group) >= 2:
                avg_level = np.mean([z['level'] for z in current_group])
                zone_type = current_group[0]['type']
                timeframes = [z['timeframe'] for z in current_group]
                total_score = sum(z['score'] for z in current_group)
                approaches = sum(z['approaches'] for z in current_group)
                age = np.mean([z['age'] for z in current_group])
                htf_present = any(tf in ['1h', '4h', '1d'] for tf in timeframes)
                ltf_present = any(tf in ['1m', '5m', '15m', '30m'] for tf in timeframes)
                relationship_score = total_score * (1.5 if htf_present and ltf_present else 1.0)
                relationship_chart.append({
                    'level': avg_level,
                    'type': zone_type,
                    'timeframes': timeframes,
                    'tf_count': len(current_group),
                    'approaches': approaches,
                    'age': age,
                    'score': relationship_score
                })
                debug_info = f"Nearby Zone Group at {avg_level:.2f} ({zone_type}): Timeframes: {', '.join(timeframes)}"
                st.session_state.zone_groups_debug[ticker].append(debug_info)
                st.session_state.trade_log.append(f"{ticker}: {debug_info}")
                if approaches == 0:
                    st.session_state.trade_log.append(f"{ticker}: Aligned zone includes fresh zone at {avg_level:.2f} ({zone_type}, Timeframes: {', '.join(timeframes)})")
            current_group = [zone]
    
    if len(current_group) >= 2:
        avg_level = np.mean([z['level'] for z in current_group])
        zone_type = current_group[0]['type']
        timeframes = [z['timeframe'] for z in current_group]
        total_score = sum(z['score'] for z in current_group)
        approaches = sum(z['approaches'] for z in current_group)
        age = np.mean([z['age'] for z in current_group])
        htf_present = any(tf in ['1h', '4h', '1d'] for tf in timeframes)
        ltf_present = any(tf in ['1m', '5m', '15m', '30m'] for tf in timeframes)
        relationship_score = total_score * (1.5 if htf_present and ltf_present else 1.0)
        relationship_chart.append({
            'level': avg_level,
            'type': zone_type,
            'timeframes': timeframes,
            'tf_count': len(current_group),
            'approaches': approaches,
            'age': age,
            'score': relationship_score
        })
        debug_info = f"Nearby Zone Group at {avg_level:.2f} ({zone_type}): Timeframes: {', '.join(timeframes)}"
        st.session_state.zone_groups_debug[ticker].append(debug_info)
        st.session_state.trade_log.append(f"{ticker}: {debug_info}")
        if approaches == 0:
            st.session_state.trade_log.append(f"{ticker}: Aligned zone includes fresh zone at {avg_level:.2f} ({zone_type}, Timeframes: {', '.join(timeframes)})")
    
    st.session_state.trade_log.append(f"{ticker}: Relationship chart created with {len(relationship_chart)} zones")
    return sorted(relationship_chart, key=lambda x: x['score'], reverse=True)

# Generate trade recommendation
def generate_trade_recommendation(dfs, aligned_zones, symbol):
    if not aligned_zones or symbol not in aligned_zones or not aligned_zones[symbol]:
        st.session_state.trade_log.append(f"No aligned zones available for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No aligned zones found", "trade": None}
    
    latest_df = dfs[symbol][3]  # Lowest timeframe
    if latest_df is None or latest_df.empty:
        st.session_state.trade_log.append(f"No data for lowest timeframe for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No data for lowest timeframe", "trade": None}
    
    current_price = latest_df['close'].iloc[-1]
    latest_candle = latest_df.iloc[-1]
    is_bullish = latest_candle['close'] > latest_candle['open']
    is_bearish = latest_candle['close'] < latest_candle['open']
    
    strong_zones = [zone for zone in aligned_zones[symbol] if zone['tf_count'] >= 2]
    if not strong_zones:
        st.session_state.trade_log.append(f"No zones found in 2+ timeframes for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No zones aligned across 2+ timeframes", "trade": None}
    
    zone_clusters = []
    strong_zones.sort(key=lambda x: x['level'])
    current_cluster = []
    for zone in strong_zones:
        if not current_cluster or abs(zone['level'] - current_cluster[0]['level']) / current_cluster[0]['level'] <= 0.01:
            current_cluster.append(zone)
        else:
            if current_cluster:
                avg_level = np.mean([z['level'] for z in current_cluster])
                zone_type = current_cluster[0]['type']
                total_approaches = sum(z['approaches'] for z in current_cluster)
                total_tf_count = sum(z['tf_count'] for z in current_cluster)
                timeframes = list(set([tf for z in current_cluster for tf in z['timeframes']]))
                htf_present = any(tf in ['1h', '4h', '1d'] for tf in timeframes)
                total_score = sum(z['score'] for z in current_cluster)
                zone_clusters.append({
                    'level': avg_level,
                    'type': zone_type,
                    'timeframes': timeframes,
                    'tf_count': total_tf_count,
                    'approaches': total_approaches,
                    'score': total_score,
                    'htf_present': htf_present
                })
            current_cluster = [zone]
    
    if current_cluster:
        avg_level = np.mean([z['level'] for z in current_cluster])
        zone_type = current_cluster[0]['type']
        total_approaches = sum(z['approaches'] for z in current_cluster)
        total_tf_count = sum(z['tf_count'] for z in current_cluster)
        timeframes = list(set([tf for z in current_cluster for tf in z['timeframes']]))
        htf_present = any(tf in ['1h', '4h', '1d'] for tf in timeframes)
        total_score = sum(z['score'] for z in current_cluster)
        zone_clusters.append({
            'level': avg_level,
            'type': zone_type,
            'timeframes': timeframes,
            'tf_count': total_tf_count,
            'approaches': total_approaches,
            'score': total_score,
            'htf_present': htf_present
        })
    
    proposed_trade = None
    for cluster in sorted(zone_clusters, key=lambda x: x['score'], reverse=True)[:3]:
        zone_level = cluster['level']
        zone_type = cluster['type']
        proximity = abs(current_price - zone_level) / current_price
        htf_present = cluster['htf_present']
        
        confidence = min(cluster['score'] / 50, 1.0) * 100
        confidence *= 1.5 if cluster['tf_count'] >= 4 else 1.3 if cluster['tf_count'] == 3 else 1.1
        confidence *= 1.1 * (1 + cluster['approaches'] / 10)
        if htf_present:
            confidence *= 1.3
        
        if confidence < 50:
            continue
        
        if zone_type == 'demand' and proximity <= 0.02 and is_bullish:
            entry = zone_level
            stop_loss = zone_level * 0.995
            next_supply = [z['level'] for z in strong_zones if z['type'] == 'supply' and z['level'] > zone_level]
            target = min(next_supply) if next_supply else zone_level * 1.02
            risk = entry - stop_loss
            reward = target - entry
            risk_reward = reward / risk if risk > 0 else 0
            if risk_reward >= 2:
                proposed_trade = {
                    'type': 'buy',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'timeframe': timeframes_list[3]
                }
                return {
                    "signal": "Buy",
                    "confidence": min(confidence, 100),
                    "details": f"Buy at demand zone cluster at {zone_level:.2f} ({cluster['tf_count']} TFs: {', '.join(cluster['timeframes'])}). Retests: {cluster['approaches']}. Target: {target:.2f}, Stop Loss: {stop_loss:.2f}, R:R: {risk_reward:.2f}",
                    "trade": proposed_trade
                }
        elif zone_type == 'supply' and proximity <= 0.02 and is_bearish:
            entry = zone_level
            stop_loss = zone_level * 1.005
            next_demand = [z['level'] for z in strong_zones if z['type'] == 'demand' and z['level'] < zone_level]
            target = max(next_demand) if next_demand else zone_level * 0.98
            risk = stop_loss - entry
            reward = entry - target
            risk_reward = reward / risk if risk > 0 else 0
            if risk_reward >= 2:
                proposed_trade = {
                    'type': 'sell',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'timeframe': timeframes_list[3]
                }
                return {
                    "signal": "Sell",
                    "confidence": min(confidence, 100),
                    "details": f"Sell at supply zone cluster at {zone_level:.2f} ({cluster['tf_count']} TFs: {', '.join(cluster['timeframes'])}). Retests: {cluster['approaches']}. Target: {target:.2f}, Stop Loss: {stop_loss:.2f}, R:R: {risk_reward:.2f}",
                    "trade": proposed_trade
                }
        elif zone_type == 'supply' and current_price > zone_level * 1.005 and is_bullish and htf_present:
            entry = current_price
            stop_loss = zone_level * 0.995
            target = zone_level * 1.03
            risk = entry - stop_loss
            reward = target - entry
            risk_reward = reward / risk if risk > 0 else 0
            if risk_reward >= 2:
                proposed_trade = {
                    'type': 'breakout',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'timeframe': timeframes_list[3]
                }
                return {
                    "signal": "Breakout",
                    "confidence": min(confidence * 0.9, 100),
                    "details": f"Breakout above supply zone cluster at {zone_level:.2f} ({cluster['tf_count']} TFs: {', '.join(cluster['timeframes'])}). Retests: {cluster['approaches']}. Target: {target:.2f}, Stop Loss: {stop_loss:.2f}, R:R: {risk_reward:.2f}",
                    "trade": proposed_trade
                }
        elif zone_type == 'demand' and current_price < zone_level * 0.995 and is_bearish and htf_present:
            entry = current_price
            stop_loss = zone_level * 1.005
            target = zone_level * 0.97
            risk = stop_loss - entry
            reward = entry - target
            risk_reward = reward / risk if risk > 0 else 0
            if risk_reward >= 2:
                proposed_trade = {
                    'type': 'breakdown',
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'timeframe': timeframes_list[3]
                }
                return {
                    "signal": "Breakdown",
                    "confidence": min(confidence * 0.9, 100),
                    "details": f"Breakdown below demand zone cluster at {zone_level:.2f} ({cluster['tf_count']} TFs: {', '.join(cluster['timeframes'])}). Retests: {cluster['approaches']}. Target: {target:.2f}, Stop Loss: {stop_loss:.2f}, R:R: {risk_reward:.2f}",
                    "trade": proposed_trade
                }
    
    return {"signal": "Hold", "confidence": 0, "details": "No breakout or breakdown detected", "trade": None}

@st.cache_data(ttl=900)  # Cache scanner data for 15 minutes
def cached_scanner_fetch(_tickers, _periods_intervals):
    """
    A cached function to fetch all data for the scanner.
    It takes tuples for list/dict arguments to be hashable for caching.
    """
    all_data = {ticker: {} for ticker in _tickers}
    
    # Reconstruct dicts and lists from cached tuples
    periods_intervals = dict(_periods_intervals)
    timeframes_list = [item for sublist in periods_intervals.values() for item in sublist]
    periods_list = list(periods_intervals.keys())

    for ticker in _tickers:
        for period, interval in zip(periods_list, timeframes_list):
            try:
                data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
                if data.empty:
                    continue
                
                data.reset_index(inplace=True)
                data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
                data = data.rename(columns={'datetime': 'date', 'index': 'date'})

                if 'date' not in data.columns:
                    continue
                data.set_index('date', inplace=True)

                if len(data) >= 10000:
                    data = data.iloc[-10000:]
                
                all_data[ticker][(period, interval)] = data
                time.sleep(0.2)
            except Exception:
                # If a download fails, we'll get an empty dict for this ticker, which is handled later
                break 
        time.sleep(1)
    return all_data

# Scanner-specific Zone identification
def scanner_identify_zones(data):
    zones = []
    if len(data) < 2: return zones
    for i in range(1, len(data) - 1):
        body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
        total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
        if body_size_base > 0.5 * total_height_base or total_height_base == 0: continue
        body_size_follow = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_follow = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        if body_size_follow < 0.75 * total_height_follow or total_height_follow == 0: continue
        date = data.index[i].to_pydatetime()
        ist = pytz.timezone('Asia/Kolkata')
        index_tz = data.index.tz
        zone_date = ist.localize(date) if date.tzinfo is None else date.astimezone(ist)
        if data['close'].iloc[i+1] > data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'demand', 'level': data['low'].iloc[i]})
        elif data['close'].iloc[i+1] < data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'supply', 'level': data['high'].iloc[i]})
    return zones

# Scanner-specific significance calculation
def scanner_calculate_zone_significance(data, zone, timeframe_idx):
    approaches = 0 # Simplified for scanner speed, can be enhanced later
    ist = pytz.timezone('Asia/Kolkata')
    age = (datetime.now(ist) - zone['date'].astimezone(ist)).total_seconds() / (3600 * 24)
    timeframe_weight = {0: 2.0, 1: 1.5, 2: 1.2, 3: 1.0}.get(timeframe_idx, 1.0)
    score = (approaches * 10 + age * 0.1) * timeframe_weight
    return score, approaches, age

# Scanner-specific relationship chart
def scanner_build_relationship_chart(dfs, zones_list, timeframes_list, ticker, tolerance):
    all_zones = []
    for idx, (df, zones, tf) in enumerate(zip(dfs, zones_list, timeframes_list)):
        if zones and df is not None and not df.empty:
            for zone in zones:
                try:
                    score, approaches, age = scanner_calculate_zone_significance(df, zone, idx)
                    all_zones.append({'level': zone['level'], 'type': zone['type'], 'timeframe': tf, 'score': score,
                                      'approaches': approaches, 'age': age, 'date': zone['date'], 'tf_idx': idx})
                except Exception: pass
    
    if not all_zones: return []
    
    relationship_chart = []
    all_zones.sort(key=lambda x: (x['level'], x['type']))
    current_group = []
    for zone in all_zones:
        if not current_group or (zone['type'] == current_group[0]['type'] and abs(zone['level'] - current_group[0]['level']) / current_group[0]['level'] <= tolerance / 100):
            current_group.append(zone)
        else:
            if len(current_group) >= 2:
                relationship_chart.append({ 'level': np.mean([z['level'] for z in current_group]), 'type': current_group[0]['type'],
                                            'timeframes': [z['timeframe'] for z in current_group], 'tf_count': len(current_group),
                                            'approaches': sum(z['approaches'] for z in current_group), 'age': np.mean([z['age'] for z in current_group]),
                                            'score': sum(z['score'] for z in current_group) })
            current_group = [zone]
    if len(current_group) >= 2:
        relationship_chart.append({ 'level': np.mean([z['level'] for z in current_group]), 'type': current_group[0]['type'],
                                    'timeframes': [z['timeframe'] for z in current_group], 'tf_count': len(current_group),
                                    'approaches': sum(z['approaches'] for z in current_group), 'age': np.mean([z['age'] for z in current_group]),
                                    'score': sum(z['score'] for z in current_group) })
    return sorted(relationship_chart, key=lambda x: x['score'], reverse=True)

# Train the model
def train_model(instances):
    if not instances or len(instances) < 2:
        return None, None
    feature_cols = ['prev_approaches']
    df = pd.DataFrame([dict(inst['features'], outcome=1 if inst['outcome'] == 'held' else 0) for inst in instances])
    st.session_state.trade_log.append(f"Training data distribution: Held={df['outcome'].sum()}, Broke={len(df) - df['outcome'].sum()}")
    X = df[feature_cols]
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.session_state.trade_log.append(f"Model accuracy: {accuracy:.2f}")
    return model, accuracy

# Backtesting strategy
def backtest_strategy(data, zones):
    trades = []
    equity = [100000]
    position = None
    zones = sorted(zones, key=lambda x: x['level'])

    for i in range(2, len(data)):
        current_time = data.index[i]
        current_candle = data.iloc[i]
        open_zones = [z for z in zones if z['date'] < current_time]
        
        for zone in open_zones:
            zone_level = zone['level']
            zone_type = zone['type']
            is_demand = zone_type == 'demand'
            body_size = abs(current_candle['close'] - current_candle['open'])
            candle_range = current_candle['high'] - current_candle['low']
            is_strong = body_size >= 0.7 * candle_range if candle_range > 0 else False
            
            touch_condition = (current_candle['low'] >= zone_level * 0.99 and current_candle['low'] <= zone_level * 1.01) if is_demand else \
                              (current_candle['high'] >= zone_level * 0.99 and current_candle['high'] <= zone_level * 1.01)
            if touch_condition and is_strong and position is None:
                if is_demand and current_candle['close'] > current_candle['open']:
                    targets = [z['level'] for z in open_zones if z['type'] == 'supply' and z['level'] > zone_level]
                    target = min(targets) if targets else zone_level * 1.02
                    position = {
                        'type': 'bounce_long',
                        'entry_time': current_time,
                        'entry_price': zone_level,
                        'target': target,
                        'stop_loss': zone_level * 0.99,
                        'highest_price': zone_level,
                        'candle_count': 0
                    }
                elif not is_demand and current_candle['close'] < current_candle['open']:
                    targets = [z['level'] for z in open_zones if z['type'] == 'demand' and z['level'] < zone_level]
                    target = max(targets) if targets else zone_level * 0.98
                    position = {
                        'type': 'bounce_short',
                        'entry_time': current_time,
                        'entry_price': zone_level,
                        'target': target,
                        'stop_loss': zone_level * 1.01,
                        'lowest_price': zone_level,
                        'candle_count': 0
                    }

        if position:
            position['candle_count'] += 1
            if 'long' in position['type']:
                position['highest_price'] = max(position['highest_price'], current_candle['high'])
                position['stop_loss'] = max(position['stop_loss'], position['highest_price'] * 0.995)
            else:
                position['lowest_price'] = min(position['lowest_price'], current_candle['low'])
                position['stop_loss'] = min(position['stop_loss'], position['lowest_price'] * 1.005)

            exit_trade = False
            if position['type'] in ['bounce_long']:
                if current_candle['high'] >= position['target']:
                    exit_price = position['target']
                    outcome = 'win'
                    exit_type = 'Target'
                    exit_trade = True
                elif current_candle['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    outcome = 'loss'
                    exit_type = 'Stop Loss'
                    exit_trade = True
            else:
                if current_candle['low'] <= position['target']:
                    exit_price = position['target']
                    outcome = 'win'
                    exit_type = 'Target'
                    exit_trade = True
                elif current_candle['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    outcome = 'loss'
                    exit_type = 'Stop Loss'
                    exit_trade = True
            if position['candle_count'] >= 20:
                exit_price = current_candle['close']
                outcome = 'neutral'
                exit_type = 'Timeout'
                exit_trade = True
            if exit_trade:
                pl = (exit_price - position['entry_price']) * 1000 if 'long' in position['type'] else (position['entry_price'] - exit_price) * 1000
                equity.append(equity[-1] + pl)
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pl': pl,
                    'outcome': outcome,
                    'exit_type': exit_type
                })
                position = None

    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    metrics = {
        'total_return': (equity[-1] - equity[0]) / equity[0] * 100 if equity.size > 1 else 0,
        'win_rate': len([t for t in trades if t['outcome'] == 'win']) / len(trades) * 100 if trades else 0,
        'max_drawdown': max([(max(equity[:i+1]) - min(equity[:i+1])) / max(equity[:i+1]) * 100 for i in range(len(equity))] or [0]),
        'num_trades': len(trades)
    }
    return trades, metrics, equity

# Plotting function
def plot_chart(df, zones, symbol, timeframe, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, aligned_zones, tolerance, show_aligned_zones, show_fresh_zones, retest_tolerance, show_proposed_trade):
    if df is None or zones is None:
        st.session_state.trade_log.append(f"plot_chart failed for {symbol} (Timeframe: {timeframe}, Period: {period}): df or zones is None")
        return None
    fig, ax = mpf.plot(df, type='candle', style='charles', returnfig=True, figsize=(10, 5))
    ax[0].set_title(f'{symbol} Candlestick (Timeframe: {timeframe}, Period: {period})')
    ax[0].set_ylabel('Price')

    for zone in zones:
        limit_price = zone['level']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        if side == 'BUY' and not show_buy_zones:
            continue
        if side == 'SELL' and not show_sell_zones:
            continue
        color = 'blue' if side == 'BUY' else 'red'
        if show_limit_lines:
            ax[0].axhline(y=limit_price, color=color, linestyle='--', alpha=0.5, linewidth=1)
            if show_prices:
                ax[0].text(len(df) - 1, limit_price, f'{limit_price:.2f}', ha='right', va='center', fontsize=10, color=color)
        st.session_state.trade_log.append(f"Plotting non-aligned zone at {limit_price:.2f} ({side}) in {color} for {symbol} ({timeframe})")

    if show_aligned_zones:
        for az in aligned_zones.get(symbol, []):
            if az['tf_count'] < 2:
                continue
            zone_level = az['level']
            zone_type = az['type']
            side = 'BUY' if zone_type == 'demand' else 'SELL'
            if side == 'BUY' and not show_buy_zones:
                continue
            if side == 'SELL' and not show_sell_zones:
                continue
            if show_limit_lines:
                ax[0].axhline(y=zone_level, color='black', linestyle='--', alpha=0.8, linewidth=2)
                if show_prices:
                    ax[0].text(len(df) - 1, zone_level, f'{zone_level:.2f}', ha='right', va='center', fontsize=10, color='black')
                st.session_state.trade_log.append(f"Plotting aligned zone at {zone_level:.2f} ({side}) in black for {symbol} ({timeframe})")

    if show_fresh_zones:
        for fz in st.session_state.fresh_zones.get(symbol, []):
            if fz['timeframe'] != timeframe:
                continue
            zone_level = fz['level']
            zone_type = fz['type']
            side = 'BUY' if zone_type == 'demand' else 'SELL'
            if side == 'BUY' and not show_buy_zones:
                continue
            if side == 'SELL' and not show_sell_zones:
                continue
            if show_limit_lines:
                ax[0].axhline(y=zone_level, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
                if show_prices:
                    ax[0].text(len(df) - 1, zone_level, f'{zone_level:.2f}', ha='right', va='center', fontsize=10, color='green')
                st.session_state.trade_log.append(f"Plotting fresh zone at {zone_level:.2f} ({side}) in green for {symbol} ({timeframe})")

    if show_limit_lines:
        for az in aligned_zones.get(symbol, []):
            if az['tf_count'] < 2:
                continue
            zone_level = az['level']
            zone_type = az['type']
            side = 'BUY' if zone_type == 'demand' else 'SELL'
            if side == 'BUY' and not show_buy_zones:
                continue
            if side == 'SELL' and not show_sell_zones:
                continue
            retest_zone = df[
                ((df['low'] >= zone_level * (1 - retest_tolerance / 100)) & (df['low'] <= zone_level * (1 + retest_tolerance / 100))) if zone_type == 'demand'
                else ((df['high'] >= zone_level * (1 - retest_tolerance / 100)) & (df['high'] <= zone_level * (1 + retest_tolerance / 100)))
            ]
            if not retest_zone.empty:
                for idx in retest_zone.index:
                    x = df.index.get_loc(idx)
                    y = retest_zone.loc[idx, 'low' if zone_type == 'demand' else 'high']
                    ax[0].scatter(x, y, marker='', color='purple', s=50)
                    st.session_state.trade_log.append(f"Potential retest at {y:.2f} for {zone_type} zone at {zone_level:.2f} (TF: {timeframe}, {symbol})")

    if show_proposed_trade and symbol in st.session_state.proposed_trade and st.session_state.proposed_trade[symbol] and timeframe == timeframes_list[3]:
        trade = st.session_state.proposed_trade[symbol]
        x = len(df) - 1
        if trade['type'] in ['buy', 'breakout']:
            ax[0].scatter(x, trade['entry'], marker='^', color='green', s=100, label='Entry')
            ax[0].scatter(x, trade['stop_loss'], marker='v', color='blue', s=100, label='Stop Loss')
            ax[0].scatter(x, trade['target'], marker='o', color='yellow', s=100, label='Target')
            ax[0].axhline(y=trade['entry'], color='green', linestyle=':', alpha=0.6, linewidth=1)
            ax[0].axhline(y=trade['stop_loss'], color='blue', linestyle=':', alpha=0.6, linewidth=1)
            ax[0].axhline(y=trade['target'], color='yellow', linestyle=':', alpha=0.6, linewidth=1)
            ax[0].text(x, trade['entry'], f"Entry: {trade['entry']:.2f}", ha='right', va='bottom', fontsize=8, color='green')
            ax[0].text(x, trade['stop_loss'], f"SL: {trade['stop_loss']:.2f}", ha='right', va='top', fontsize=8, color='blue')
            ax[0].text(x, trade['target'], f"Target: {trade['target']:.2f}", ha='right', va='bottom', fontsize=8, color='yellow')
        else:
            ax[0].scatter(x, trade['entry'], marker='v', color='red', s=100, label='Entry')
            ax[0].scatter(x, trade['stop_loss'], marker='^', color='blue', s=100, label='Stop Loss')
            ax[0].scatter(x, trade['target'], marker='o', color='yellow', s=100, label='Target')
            ax[0].axhline(y=trade['entry'], color='red', linestyle=':', alpha=0.6, linewidth=1)
            ax[0].axhline(y=trade['stop_loss'], color='blue', linestyle=':', alpha=0.6, linewidth=1)
            ax[0].axhline(y=trade['target'], color='yellow', linestyle=':', alpha=0.6, linewidth=1)
            ax[0].text(x, trade['entry'], f"Entry: {trade['entry']:.2f}", ha='right', va='top', fontsize=8, color='red')
            ax[0].text(x, trade['stop_loss'], f"SL: {trade['stop_loss']:.2f}", ha='right', va='bottom', fontsize=8, color='blue')
            ax[0].text(x, trade['target'], f"Target: {trade['target']:.2f}", ha='right', va='top', fontsize=8, color='yellow')
        ax[0].legend()
        st.session_state.trade_log.append(f"Plotted proposed {trade['type']} trade for {symbol}: Entry={trade['entry']:.2f}, SL={trade['stop_loss']:.2f}, Target={trade['target']:.2f}")

    return fig

# Plot trade chart
def plot_trade_chart(df, zones, trades, symbol, timeframe, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, aligned_zones, show_aligned_zones, show_fresh_zones):
    if df is None or zones is None or not trades:
        st.session_state.trade_log.append(f"plot_trade_chart failed for {symbol} (Timeframe: {timeframe}, Period: {period}): df, zones, or trades invalid")
        return None
    fig, ax = mpf.plot(df, type='candle', style='charles', returnfig=True, figsize=(10, 5))
    ax[0].set_title(f'{symbol} Trades (Timeframe: {timeframe}, Period: {period})')
    ax[0].set_ylabel('Price')

    for zone in zones:
        limit_price = zone['level']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        if side == 'BUY' and not show_buy_zones:
            continue
        if side == 'SELL' and not show_sell_zones:
            continue
        color = 'blue' if side == 'BUY' else 'red'
        if show_limit_lines:
            ax[0].axhline(y=limit_price, color=color, linestyle='--', alpha=0.5, linewidth=1)

    if show_aligned_zones:
        for az in aligned_zones.get(symbol, []):
            if az['tf_count'] < 2:
                continue
            zone_level = az['level']
            zone_type = az['type']
            side = 'BUY' if zone_type == 'demand' else 'SELL'
            if side == 'BUY' and not show_buy_zones:
                continue
            if side == 'SELL' and not show_sell_zones:
                continue
            if show_limit_lines:
                ax[0].axhline(y=zone_level, color='black', linestyle='--', alpha=0.8, linewidth=2)

    if show_fresh_zones:
        for fz in st.session_state.fresh_zones.get(symbol, []):
            if fz['timeframe'] != timeframe:
                continue
            zone_level = fz['level']
            zone_type = fz['type']
            side = 'BUY' if zone_type == 'demand' else 'SELL'
            if side == 'BUY' and not show_buy_zones:
                continue
            if side == 'SELL' and not show_sell_zones:
                continue
            if show_limit_lines:
                ax[0].axhline(y=zone_level, color='green', linestyle='--', alpha=0.7, linewidth=1.5)

    entry_times = []
    entry_prices = []
    exit_times = []
    exit_prices = []
    stop_loss_times = []
    stop_loss_prices = []
    for trade in trades:
        try:
            entry_idx = df.index.get_loc(trade['entry_time'])
            exit_idx = df.index.get_loc(trade['exit_time'])
            entry_times.append(entry_idx)
            entry_prices.append(trade['entry_price'])
            if trade['exit_type'] == 'Stop Loss':
                stop_loss_times.append(exit_idx)
                stop_loss_prices.append(trade['exit_price'])
            else:
                exit_times.append(exit_idx)
                exit_prices.append(trade['exit_price'])
        except KeyError:
            continue

    if entry_times:
        ax[0].scatter(entry_times, entry_prices, marker='^', color='#2ca02c', label='Entry', s=100)
    if exit_times:
        ax[0].scatter(exit_times, exit_prices, marker='v', color='#d62728', label='Exit', s=100)
    if stop_loss_times:
        ax[0].scatter(stop_loss_times, stop_loss_prices, marker='v', color='#1f77b4', label='Stop Loss Exit', s=100)
    ax[0].legend()
    return fig

# Streamlit UI
st.set_page_config(page_title="Indian Stocks Trading Dashboard", layout="wide")
st.title("Indian Stocks Trading Dashboard")

# Guidelines
st.markdown("""
**Guidelines**: 
- Enter valid tickers (e.g., `nifty50, banknifty, RELIANCE`) separated by commas.
- Use consistent timeframes and periods (e.g., `1d` with `1mo`, `1wk` with `6mo`).
- Uncheck `NSE stocks` for indices like `nifty50` (`^NSEI`); check for stocks (e.g., `RELIANCE` → `RELIANCE.NS`).
""")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    tickers = st.text_input("Enter stock symbols", "nifty50")
    append_ns = st.checkbox("NSE stocks", value=True)
    
    st.subheader("Timeframe 1")
    timeframe_1 = st.selectbox("Timeframe 1", list(timeframes.keys()), index=7, key="tf1")
    period_1 = st.selectbox("Period 1", periods, index=4, key="p1")
    
    st.subheader("Timeframe 2")
    timeframe_2 = st.selectbox("Timeframe 2", list(timeframes.keys()), index=6, key="tf2")
    period_2 = st.selectbox("Period 2", periods, index=3, key="p2")
    
    st.subheader("Timeframe 3")
    timeframe_3 = st.selectbox("Timeframe 3", list(timeframes.keys()), index=4, key="tf3")
    period_3 = st.selectbox("Period 3", periods, index=1, key="p3")
    
    st.subheader("Timeframe 4")
    timeframe_4 = st.selectbox("Timeframe 4", list(timeframes.keys()), index=1, key="tf4")
    period_4 = st.selectbox("Period 4", periods, index=0, key="p4")
    
    st.subheader("Zone Grouping Tolerance")
    tolerance = st.slider("Price Tolerance for Nearby Zones (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    
    st.subheader("Retest Detection Tolerance")
    retest_tolerance = st.slider("Price Tolerance for Retest Detection (%)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    
    show_buy_zones = st.checkbox("Show Demand Zones", value=True)
    show_sell_zones = st.checkbox("Show Supply Zones", value=True)
    show_limit_lines = st.checkbox("Show Limit Lines", value=True)
    show_prices = st.checkbox("Show Zone Prices", value=True)
    show_aligned_zones = st.checkbox("Show Aligned Zones", value=True)
    show_fresh_zones = st.checkbox("Show Fresh Zones", value=True)
    show_proposed_trade = st.checkbox("Show Proposed Trade", value=True)
    refresh_button = st.button("Refresh Data")
    backtest_button = st.button("Run Backtest")

# Process tickers
st.session_state.trade_log.append(f"Raw ticker input: {tickers}")
tickers_clean = tickers.strip()
if not tickers_clean:
    tickers_clean = "nifty50"
    st.session_state.trade_log.append("No valid tickers provided, defaulting to 'nifty50'")
entered_tickers = [ticker.strip().lower() for ticker in tickers_clean.split(",") if ticker.strip()]
if not entered_tickers:
    entered_tickers = ["nifty50"]
    st.session_state.trade_log.append("No valid tickers parsed, defaulting to ['nifty50']")
mapped_tickers = [ticker_mapping.get(ticker, ticker) for ticker in entered_tickers]
final_ticker_list = [symbol.upper() + ".NS" if append_ns and not symbol.startswith('^') else symbol.upper() for symbol in mapped_tickers if symbol]
st.session_state.trade_log.append(f"Final ticker list: {final_ticker_list}")

# Validate timeframes
timeframes_list = [timeframe_1, timeframe_2, timeframe_3, timeframe_4]
if isinstance(timeframes_list, tuple):
    timeframes_list = list(timeframes_list)
    st.session_state.trade_log.append("Converted timeframes_list from tuple to list")
st.session_state.trade_log.append(f"Timeframes list: {timeframes_list}")
periods_list = [period_1, period_2, period_3, period_4]
periods_intervals = {periods_list[i]: [timeframes_list[i]] for i in range(4)}
st.session_state.trade_log.append(f"Periods intervals: {periods_intervals}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab_scanner = st.tabs(["Trade Dashboard", "Detailed Charts", "Backtest", "Trade Charts", "Zones", "Scanner"])

# Trade Dashboard
with tab1:
    if refresh_button:
        # Reset state
        st.session_state.plot_data_ready = False
        st.session_state.dfs = {ticker: [None] * 4 for ticker in final_ticker_list}
        st.session_state.zones_list = {ticker: [None] * 4 for ticker in final_ticker_list}
        st.session_state.aligned_zones = {ticker: [] for ticker in final_ticker_list}
        st.session_state.fresh_zones = {ticker: [] for ticker in final_ticker_list}
        st.session_state.zone_groups_debug = {ticker: [] for ticker in final_ticker_list}
        st.session_state.all_zones_debug = {ticker: [] for ticker in final_ticker_list}
        st.session_state.recommendation = {}
        st.session_state.proposed_trade = {}
        
        all_data = fetch_and_process_data(final_ticker_list, periods_intervals)
        for ticker in final_ticker_list:
            data_fetched = False
            if ticker in all_data:
                for idx, (tf, period) in enumerate(zip(timeframes_list, periods_list)):
                    if (period, tf) in all_data[ticker]:
                        df = all_data[ticker][(period, tf)]
                        st.session_state.dfs[ticker][idx] = df
                        st.session_state.zones_list[ticker][idx] = identify_zones(df)
                        data_fetched = True
                        st.session_state.trade_log.append(f"Found {len(st.session_state.zones_list[ticker][idx])} zones for {ticker} (Timeframe: {tf}, Period: {period})")
                    else:
                        st.session_state.trade_log.append(f"No data found for {ticker} (Timeframe: {tf}, Period: {period})")
            
            if not data_fetched:
                st.error(f"No valid data fetched for {ticker}. Please check ticker or timeframe settings.")
            else:
                try:
                    st.session_state.aligned_zones[ticker] = build_relationship_chart(
                        st.session_state.dfs, st.session_state.zones_list, timeframes_list, ticker, tolerance
                    )
                    recommendation = generate_trade_recommendation(
                        {ticker: st.session_state.dfs[ticker]}, st.session_state.aligned_zones, ticker
                    )
                    st.session_state.recommendation[ticker] = recommendation
                    st.session_state.proposed_trade[ticker] = recommendation.get('trade')
                    st.session_state.trade_log.append(f"Found {len(st.session_state.aligned_zones[ticker])} aligned zones for {ticker}. Recommendation: {recommendation['signal']}")
                except Exception as e:
                    st.session_state.trade_log.append(f"Error generating recommendation for {ticker}: {str(e)}")
                    st.error(f"Error generating recommendation for {ticker}: {str(e)}")
        
        
        st.session_state.plot_data_ready = True
    
    st.header("Trade Dashboard")
    selected_ticker = st.selectbox("Select Ticker", final_ticker_list, key="ticker_select")
    
    if st.session_state.plot_data_ready and selected_ticker in st.session_state.recommendation:
        rec = st.session_state.recommendation[selected_ticker]
        st.subheader(f"Trade Recommendation for {selected_ticker}")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Signal", rec['signal'], delta=None)
            st.metric("Confidence", f"{rec['confidence']:.1f}%")
        with col2:
            st.write("**Details**")
            st.write(rec['details'])
        
        st.subheader("Proposed Trade")
        if st.session_state.proposed_trade.get(selected_ticker):
            trade = st.session_state.proposed_trade[selected_ticker]
            trade_df = pd.DataFrame([{
                'Type': trade['type'].capitalize(),
                'Entry': round(trade['entry'], 2),
                'Stop Loss': round(trade['stop_loss'], 2),
                'Target': round(trade['target'], 2),
                'Risk-Reward': round(trade['risk_reward'], 2),
                'Timeframe': trade['timeframe']
            }])
            # Apply conditional styling
            def style_trade(row):
                color = 'green' if row['Type'].lower() in ['buy', 'breakout'] else 'red'
                return [f'color: {color}' if col == 'Type' else '' for col in row.index]
            st.dataframe(trade_df.style.apply(style_trade, axis=1))
        else:
            st.write("No proposed trade available.")
        
        # Plot chart for lowest timeframe
        df = st.session_state.dfs[selected_ticker][3]
        zones = st.session_state.zones_list[selected_ticker][3]
        tf = timeframes_list[3]
        period = periods_list[3]
        if df is not None and zones is not None:
            st.subheader(f"Chart for {selected_ticker} (Timeframe: {tf}, Period: {period})")
            fig = plot_chart(df, zones, selected_ticker, tf, period, show_buy_zones, show_sell_zones, 
                            show_limit_lines, show_prices, st.session_state.aligned_zones, 
                            tolerance, show_aligned_zones, show_fresh_zones, retest_tolerance,
                            show_proposed_trade)
            if fig:
                st.pyplot(fig)
                st.session_state.trade_log.append(f"Chart plotted for {selected_ticker} (Timeframe: {tf})")
                plt.close(fig)
        
        with st.expander("View Zone Details"):
            st.subheader(f"Fresh Zones (0 Retests) for {selected_ticker}")
            fresh_zones_df = pd.DataFrame(st.session_state.fresh_zones.get(selected_ticker, []))
            if not fresh_zones_df.empty:
                fresh_zones_df['level'] = fresh_zones_df['level'].round(2)
                fresh_zones_df['age'] = fresh_zones_df['age'].round(1)
                fresh_zones_df = fresh_zones_df[['level', 'type', 'timeframe', 'age']]
                fresh_zones_df.columns = ['Price Level', 'Type', 'Timeframe', 'Age (days)']
                st.dataframe(fresh_zones_df)
            else:
                st.write("No fresh zones found.")
            
            st.subheader(f"Aligned Zones (2+ Timeframes) for {selected_ticker}")
            zones_data = st.session_state.aligned_zones.get(selected_ticker, [])
            if zones_data:
                zones_df = pd.DataFrame(zones_data)
                zones_df['timeframes'] = zones_df['timeframes'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
                zones_df['level'] = zones_df['level'].round(2)
                zones_df['age'] = zones_df['age'].round(1)
                zones_df = zones_df[['level', 'type', 'timeframes', 'tf_count', 'approaches', 'age', 'score']]
                zones_df.columns = ['Price Level', 'Type', 'Timeframes', 'TF Count', 'Retests', 'Age (days)', 'Score']
                st.dataframe(zones_df)
            else:
                st.write("No aligned zones found.")
    else:
        st.write("Click 'Refresh Data' to generate trade recommendations.")

# Detailed Charts Tab
with tab2:
    if st.session_state.plot_data_ready:
        for ticker in final_ticker_list:
            if ticker in st.session_state.dfs and ticker in st.session_state.zones_list:
                st.subheader(f"Charts for {ticker}")
                cols = st.columns(2)
                for idx, (df, zones, tf, period) in enumerate(zip(st.session_state.dfs[ticker], st.session_state.zones_list[ticker], timeframes_list, periods_list)):
                    if df is not None and zones is not None:
                        with cols[idx % 2]:
                            fig = plot_chart(df, zones, ticker, tf, period, show_buy_zones, show_sell_zones, 
                                            show_limit_lines, show_prices, st.session_state.aligned_zones, 
                                            tolerance, show_aligned_zones, show_fresh_zones, retest_tolerance,
                                            show_proposed_trade)
                            if fig:
                                st.pyplot(fig)
                                st.session_state.trade_log.append(f"Detailed chart plotted for {ticker} (Timeframe: {tf})")
                                plt.close(fig)
            else:
                st.warning(f"Could not display charts for {ticker} as data fetching may have failed.")
    else:
        st.write("Click 'Refresh Data' to view charts.")

# Backtest Tab
with tab3:
    if backtest_button:
        st.session_state.backtest_data_ready = False
        st.session_state.trades_list = {ticker: [[] for _ in range(4)] for ticker in final_ticker_list}
        st.session_state.metrics_list = {ticker: [{} for _ in range(4)] for ticker in final_ticker_list}
        st.session_state.equity_list = {ticker: [[100000] for _ in range(4)] for ticker in final_ticker_list}
        
        for ticker in final_ticker_list:
            for idx, (df, zones, tf, period) in enumerate(zip(st.session_state.dfs[ticker], st.session_state.zones_list[ticker], timeframes_list, periods_list)):
                if df is None or zones is None:
                    continue
                filtered_zones = [z for z in zones if (z['type'] == 'demand' and show_buy_zones) or (z['type'] == 'supply' and show_sell_zones)]
                trades, metrics, equity = backtest_strategy(df, filtered_zones)
                st.session_state.trades_list[ticker][idx] = trades
                st.session_state.metrics_list[ticker][idx] = metrics
                st.session_state.equity_list[ticker][idx] = equity
                st.session_state.trade_log.append(f"Backtest completed for {ticker} (Timeframe {idx+1}: {tf}): {metrics['num_trades']} trades")
        
        st.session_state.backtest_data_ready = True
    
    if st.session_state.backtest_data_ready:
        for ticker in final_ticker_list:
            st.subheader(f"Backtest Results for {ticker}")
            cols = st.columns(2)
            for idx, (metrics, equity, trades, tf, period) in enumerate(zip(st.session_state.metrics_list[ticker], st.session_state.equity_list[ticker], st.session_state.trades_list[ticker], timeframes_list, periods_list)):
                if metrics:
                    with cols[idx % 2]:
                        st.write(f"**Timeframe: {tf}, Period: {period}**")
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                            st.metric("Number of Trades", metrics.get('num_trades', 0))
                        with col4:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.plot(equity, color='blue', label='Equity')
                            ax.set_title(f"Equity Curve (TF {idx+1})")
                            ax.set_xlabel("Trade")
                            ax.set_ylabel("Equity ($)")
                            ax.grid(True)
                            ax.legend()
                            st.pyplot(fig)
                            plt.close(fig)
                            if trades:
                                trade_df = pd.DataFrame(trades)
                                trade_df['entry_time'] = trade_df['entry_time'].astype(str)
                                trade_df['exit_time'] = trade_df['exit_time'].astype(str)
                                st.dataframe(trade_df[['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price', 'pl', 'exit_type']])
    else:
        st.write("Click 'Run Backtest' to view results.")

# Trade Charts Tab
with tab4:
    if st.session_state.backtest_data_ready and any(any(trades for trades in tls.values()) for tls in st.session_state.trades_list.values()):
        for ticker in final_ticker_list:
            st.subheader(f"Trade Charts for {ticker}")
            cols = st.columns(2)
            for idx, (df, zones, trades, tf, period) in enumerate(zip(st.session_state.dfs[ticker], st.session_state.zones_list[ticker], st.session_state.trades_list[ticker], timeframes_list, periods_list)):
                if df is not None and zones is not None and trades:
                    with cols[idx % 2]:
                        fig = plot_trade_chart(df, zones, trades, ticker, tf, period, show_buy_zones, show_sell_zones, 
                                              show_limit_lines, show_prices, st.session_state.aligned_zones, 
                                              show_aligned_zones, show_fresh_zones)
                        if fig:
                            st.pyplot(fig)
                            st.session_state.trade_log.append(f"Trade chart plotted for {ticker} (Timeframe {idx+1}: {tf})")
                            plt.close(fig)
    else:
        st.write("Run a backtest to view trade charts.")

# Zones Tab
with tab5:
    if st.session_state.plot_data_ready:
        for ticker in final_ticker_list:
            st.subheader(f"Zones for {ticker}")
            st.write("**Fresh Zones (0 Retests)**")
            fresh_zones_df = pd.DataFrame(st.session_state.fresh_zones.get(ticker, []))
            if not fresh_zones_df.empty:
                fresh_zones_df['level'] = fresh_zones_df['level'].round(2)
                fresh_zones_df['age'] = fresh_zones_df['age'].round(1)
                fresh_zones_df = fresh_zones_df[['level', 'type', 'timeframe', 'age']]
                fresh_zones_df.columns = ['Price Level', 'Type', 'Timeframe', 'Age (days)']
                st.dataframe(fresh_zones_df)
            else:
                st.write("No fresh zones found.")
            
            st.write("**Aligned Zones (2+ Timeframes)**")
            zones_data = st.session_state.aligned_zones.get(ticker, [])
            if zones_data:
                zones_df = pd.DataFrame(zones_data)
                zones_df['timeframes'] = zones_df['timeframes'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
                zones_df['level'] = zones_df['level'].round(2)
                zones_df['age'] = zones_df['age'].round(1)
                zones_df = zones_df[['level', 'type', 'timeframes', 'tf_count', 'approaches', 'age', 'score']]
                zones_df.columns = ['Price Level', 'Type', 'Timeframes', 'TF Count', 'Retests', 'Age (days)', 'Score']
                st.dataframe(zones_df)
            else:
                st.write("No aligned zones found.")
            
            with st.expander("Debug: All Zones by Timeframe"):
                all_zones_df = pd.DataFrame(st.session_state.all_zones_debug.get(ticker, []))
                if not all_zones_df.empty:
                    all_zones_df['level'] = all_zones_df['level'].round(2)
                    all_zones_df['age'] = all_zones_df['age'].round(1)
                    all_zones_df = all_zones_df[['level', 'type', 'timeframe', 'approaches', 'age']]
                    all_zones_df.columns = ['Price Level', 'Type', 'Timeframe', 'Retests', 'Age (days)']
                    st.dataframe(all_zones_df)
                else:
                    st.write("No zones found for debugging.")
    else:
        st.write("Click 'Refresh Data' to view zones.")

# Scanner Tab
with tab_scanner:
    scanner_tab1, scanner_tab2, scanner_tab3 = st.tabs(["Scan", "Manage Watch Lists", "Results"])

    # This function remains here as it's the core processing logic for scans
    def process_scan_results(scanner_tickers, all_scanner_data, periods_intervals, timeframes_list, tolerance):
        all_trades = []
        analysis_progress = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(scanner_tickers):
            status_text.text(f"Analyzing {ticker} ({i+1}/{len(scanner_tickers)})...")
            try:
                ticker_dfs = [None] * 4
                if ticker in all_scanner_data and all_scanner_data[ticker]:
                    main_periods_list = [period_1, period_2, period_3, period_4]
                    main_timeframes_list = [timeframe_1, timeframe_2, timeframe_3, timeframe_4]
                    for idx, (period, tf) in enumerate(zip(main_periods_list, main_timeframes_list)):
                        if (period, tf) in all_scanner_data[ticker]:
                            ticker_dfs[idx] = all_scanner_data[ticker][(period, tf)]
                
                if not any(df is not None for df in ticker_dfs):
                    continue

                ticker_zones_list = [identify_zones(df) if df is not None else [] for df in ticker_dfs]
                temp_dfs = {ticker: ticker_dfs}
                temp_zones_list = {ticker: ticker_zones_list}
                aligned_zones_list = build_relationship_chart(temp_dfs, temp_zones_list, timeframes_list, ticker, tolerance)
                temp_aligned_zones = {ticker: aligned_zones_list}
                recommendation = generate_trade_recommendation(temp_dfs, temp_aligned_zones, ticker)
                
                if recommendation and recommendation.get('trade'):
                    trade_info = recommendation['trade']
                    all_trades.append({ 'Ticker': ticker, 'Signal': recommendation['signal'], 'Entry': trade_info['entry'],
                                        'Stop Loss': trade_info['stop_loss'], 'Target': trade_info['target'],
                                        'Risk-Reward': trade_info['risk_reward'], 'Confidence': recommendation['confidence']})
            except Exception as e:
                st.error(f"Failed to analyze {ticker}: {e}")
            
            analysis_progress.progress((i + 1) / len(scanner_tickers))
        
        status_text.text("Analysis complete.")
        return sorted(all_trades, key=lambda x: x['Risk-Reward'], reverse=True)

    # --- Scan Tab ---
    with scanner_tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.header("Market Watch Scan")
            market_watches = load_market_watches()
            watch_list_name_scan = st.selectbox("Select Market Watch to Scan", options=list(market_watches.keys()), key="scan_watchlist")
            
            st.write(f"**Tickers in {watch_list_name_scan}:**")
            if market_watches[watch_list_name_scan]:
                st.text(', '.join(market_watches[watch_list_name_scan]))
            else:
                st.text("No tickers in this list.")

            if st.button(f"Scan {watch_list_name_scan}"):
                raw_tickers = market_watches[watch_list_name_scan]
                scanner_tickers = [t + ".NS" if not t.endswith(".NS") else t for t in raw_tickers]
                if not scanner_tickers:
                    st.warning("No tickers in the selected market watch to scan.")
                else:
                    with st.spinner("Fetching data (this may be instant if cached)..."):
                        tickers_tuple = tuple(scanner_tickers)
                        periods_intervals_tuple = tuple(sorted(periods_intervals.items()))
                        all_scanner_data = cached_scanner_fetch(tickers_tuple, periods_intervals_tuple)
                    
                    st.session_state.scanner_results = process_scan_results(scanner_tickers, all_scanner_data, periods_intervals, timeframes_list, tolerance)
                    st.success("Scan complete! Check the 'Results' tab.")

        with col2:
            st.header("Manual Scan")
            scanner_stock_list_input = st.text_area("Enter stock tickers (one per line or comma-separated):", "RELIANCE, TCS, HDFCBANK, INFY", height=155)
            if st.button("Start Manual Scan"):
                raw_tickers = [ticker.strip().upper() for ticker in scanner_stock_list_input.replace(",", "\n").split() if ticker.strip()]
                scanner_tickers = [t + ".NS" if not t.endswith(".NS") else t for t in raw_tickers]
                if not scanner_tickers:
                    st.warning("Please enter at least one stock ticker to scan.")
                else:
                    with st.spinner("Fetching data (this may be instant if cached)..."):
                        tickers_tuple = tuple(scanner_tickers)
                        periods_intervals_tuple = tuple(sorted(periods_intervals.items()))
                        all_scanner_data = cached_scanner_fetch(tickers_tuple, periods_intervals_tuple)

                    st.session_state.scanner_results = process_scan_results(scanner_tickers, all_scanner_data, periods_intervals, timeframes_list, tolerance)
                    st.success("Scan complete! Check the 'Results' tab.")

    # --- Manage Watch Lists Tab ---
    with scanner_tab2:
        st.header("Manage Watch Lists")
        market_watches_manage = load_market_watches()
        watch_list_name_manage = st.selectbox("Select Market Watch to Manage", options=list(market_watches_manage.keys()), key="manage_watchlist")

        st.subheader(f"Tickers in {watch_list_name_manage}")
        if market_watches_manage[watch_list_name_manage]:
            st.dataframe(pd.DataFrame(market_watches_manage[watch_list_name_manage], columns=["Ticker"]), use_container_width=True)
        else:
            st.write("No tickers in this market watch.")
        
        st.divider()
        
        # --- Nifty 500 Stock Loader ---
        @st.cache_data(ttl=3600)
        def load_nifty500_stocks():
            APP_DIR = os.path.dirname(os.path.abspath(__file__))
            NIFTY500_CSV_PATH = os.path.join(APP_DIR, "nifty500_9oct.csv")
            if not os.path.exists(NIFTY500_CSV_PATH):
                st.error(f"Error: Stock list CSV not found. Please ensure 'nifty500_9oct.csv' is in the app directory.")
                return []
            try:
                df = pd.read_csv(NIFTY500_CSV_PATH)
                if 'Symbol' not in df.columns:
                    st.error(f"Error: '{NIFTY500_CSV_PATH}' must contain a 'Symbol' column.")
                    return []
                return df['Symbol'].tolist()
            except Exception as e:
                st.error(f"Failed to load stock list from '{NIFTY500_CSV_PATH}': {e}")
                return []

        nifty500_stocks = load_nifty500_stocks()
        MAX_WATCHLIST_SIZE = 100

        col1_manage, col2_manage = st.columns(2)

        with col1_manage:
            st.subheader("Add Tickers")
            st.info(f"A watch list can hold a maximum of {MAX_WATCHLIST_SIZE} tickers.")
            stocks_to_add = st.multiselect("Search and Select Tickers to Add", options=nifty500_stocks, placeholder="Type to search for stocks...")
            if st.button("Add Selected Stocks"):
                if not stocks_to_add:
                    st.warning("Please select at least one stock to add.")
                else:
                    current_size = len(market_watches_manage[watch_list_name_manage])
                    if current_size + len(stocks_to_add) > MAX_WATCHLIST_SIZE:
                        st.error(f"Cannot add {len(stocks_to_add)} stocks. This would exceed the limit of {MAX_WATCHLIST_SIZE} stocks.")
                    else:
                        added_count, skipped_count = 0, 0
                        for ticker in stocks_to_add:
                            if ticker not in market_watches_manage[watch_list_name_manage]:
                                market_watches_manage[watch_list_name_manage].append(ticker)
                                added_count += 1
                            else:
                                skipped_count += 1
                        if added_count > 0:
                            save_market_watches(market_watches_manage)
                            st.success(f"Added {added_count} new ticker(s).")
                        if skipped_count > 0:
                            st.info(f"Skipped {skipped_count} ticker(s) that were already in the list.")
                        st.rerun()
        
        with col2_manage:
            st.subheader("Delete Tickers")
            if market_watches_manage[watch_list_name_manage]:
                ticker_to_delete = st.selectbox("Select Ticker to Delete", options=[""] + sorted(market_watches_manage[watch_list_name_manage]))
                if st.button("Delete Ticker"):
                    if ticker_to_delete:
                        market_watches_manage[watch_list_name_manage].remove(ticker_to_delete)
                        save_market_watches(market_watches_manage)
                        st.success(f"Removed {ticker_to_delete} from {watch_list_name_manage}.")
                        st.rerun()
                    else:
                        st.warning("Please select a ticker to delete.")
            else:
                st.write("No tickers to delete.")

    # --- Results Tab ---
    with scanner_tab3:
        st.header("Market Scanner Results")
        if 'scanner_results' in st.session_state and st.session_state.scanner_results:
            st.write("Showing results from the last scan, ranked by Risk-Reward ratio.")
            scanner_df = pd.DataFrame(st.session_state.scanner_results)
            scanner_df['Entry'] = scanner_df['Entry'].round(2)
            scanner_df['Stop Loss'] = scanner_df['Stop Loss'].round(2)
            scanner_df['Target'] = scanner_df['Target'].round(2)
            scanner_df['Risk-Reward'] = scanner_df['Risk-Reward'].round(2)
            scanner_df['Confidence'] = scanner_df['Confidence'].round(1).astype(str) + '%'
            st.dataframe(scanner_df, use_container_width=True)
        else:
            st.info("No scan has been run yet, or no trade opportunities were found in the last scan.")

# Trade Log
st.header("Trade Log")
st.text_area("Log", "\n".join(st.session_state.trade_log[-20:]), height=200, disabled=True)

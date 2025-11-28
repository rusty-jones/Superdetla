import streamlit as st
import numpy as np
import pytz
from datetime import datetime
import time

def identify_zones(data):
    zones = []
    if len(data) < 2:
        if 'trade_log' in st.session_state:
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
    if 'trade_log' in st.session_state:
        st.session_state.trade_log.append(f"Identified {len(zones)} zones for data length {len(data)}")
    return zones

def find_approaches_and_labels(data, zones):
    instances = []
    if not zones or len(data) < 2:
        if 'trade_log' in st.session_state:
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
    if 'trade_log' in st.session_state:
        st.session_state.trade_log.append(f"Found {len(instances)} approach instances for {len(zones)} zones")
    return instances

def calculate_zone_significance(data, zone, timeframe_idx):
    approaches = len(find_approaches_and_labels(data, [zone]))
    ist = pytz.timezone('Asia/Kolkata')
    age = (datetime.now(ist) - zone['date'].astimezone(ist)).total_seconds() / (3600 * 24)
    timeframe_weight = {0: 2.0, 1: 1.5, 2: 1.2, 3: 1.0}.get(timeframe_idx, 1.0)
    score = (approaches * 10 + age * 0.1) * timeframe_weight
    if 'trade_log' in st.session_state:
        st.session_state.trade_log.append(f"Zone at {zone['level']:.2f} ({zone['type']}, TF {timeframe_idx+1}): approaches={approaches}, age={age:.1f} days, score={score:.2f}")
    return score, approaches, age

def build_relationship_chart(dfs, zones_list, timeframes_list, ticker, tolerance):
    relationship_chart = []
    all_zones = []
    if 'fresh_zones' not in st.session_state:
        st.session_state.fresh_zones = {}
    st.session_state.fresh_zones[ticker] = []
    if 'zone_groups_debug' not in st.session_state:
        st.session_state.zone_groups_debug = {}
    st.session_state.zone_groups_debug[ticker] = []
    if 'all_zones_debug' not in st.session_state:
        st.session_state.all_zones_debug = {}
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
                        if 'trade_log' in st.session_state:
                            st.session_state.trade_log.append(f"Added fresh zone at {zone['level']:.2f} ({zone['type']}) in {tf} for {ticker}")
                except Exception as e:
                    if 'trade_log' in st.session_state:
                        st.session_state.trade_log.append(f"Error calculating significance for {ticker} (TF {tf}): {str(e)}")
                time.sleep(0.01)
    
    if not all_zones:
        if 'trade_log' in st.session_state:
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
                if 'trade_log' in st.session_state:
                    st.session_state.trade_log.append(f"{ticker}: {debug_info}")
                if approaches == 0:
                    if 'trade_log' in st.session_state:
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
        if 'trade_log' in st.session_state:
            st.session_state.trade_log.append(f"{ticker}: {debug_info}")
        if approaches == 0:
            if 'trade_log' in st.session_state:
                st.session_state.trade_log.append(f"{ticker}: Aligned zone includes fresh zone at {avg_level:.2f} ({zone_type}, Timeframes: {', '.join(timeframes)})")
    
    if 'trade_log' in st.session_state:
        st.session_state.trade_log.append(f"{ticker}: Relationship chart created with {len(relationship_chart)} zones")
    return sorted(relationship_chart, key=lambda x: x['score'], reverse=True)

def generate_trade_recommendation(dfs, aligned_zones, symbol, timeframes_list):
    if not aligned_zones or symbol not in aligned_zones or not aligned_zones[symbol]:
        if 'trade_log' in st.session_state:
            st.session_state.trade_log.append(f"No aligned zones available for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No aligned zones found", "trade": None}
    
    latest_df = dfs[symbol][3]
    if latest_df is None or latest_df.empty:
        if 'trade_log' in st.session_state:
            st.session_state.trade_log.append(f"No data for lowest timeframe for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No data for lowest timeframe", "trade": None}
    
    current_price = latest_df['close'].iloc[-1]
    latest_candle = latest_df.iloc[-1]
    is_bullish = latest_candle['close'] > latest_candle['open']
    is_bearish = latest_candle['close'] < latest_candle['open']
    
    strong_zones = [zone for zone in aligned_zones[symbol] if zone['tf_count'] >= 2]
    if not strong_zones:
        if 'trade_log' in st.session_state:
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

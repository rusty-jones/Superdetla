import numpy as np
from data_handler import fetch_and_process_data
from config import ticker_mapping

def identify_zones(df, interval, trade_log):
    try:
        zones = []
        window = 3 if interval == '5m' else 1
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window + 1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window + 1)):
                zones.append({'date': df.index[i], 'type': 'demand', 'level': df['low'].iloc[i]})
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window + 1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window + 1)):
                zones.append({'date': df.index[i], 'type': 'supply', 'level': df['high'].iloc[i]})
        trade_log.append(f"Identified {len(zones)} zones for interval {interval}")
        return zones
    except Exception as e:
        trade_log.append(f"Error identifying zones for interval {interval}: {str(e)}")
        return []

def identify_super_zones(ticker, trade_log):
    try:
        super_zones = []
        timeframe_configs = [
            {'period': '1y', 'interval': '1wk'},
            {'period': '6mo', 'interval': '1wk'},
            {'period': '6mo', 'interval': '1d'},
            {'period': '3mo', 'interval': '1d'},
            {'period': '1mo', 'interval': '1h'},
            {'period': '1mo', 'interval': '30m'},
            {'period': '5d', 'interval': '15m'},
            {'period': '1d', 'interval': '5m'},
            {'period': '1d', 'interval': '1m'}
        ]
        
        all_zones = []
        mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
        trade_log.append(f"Identifying super zones for {ticker} (mapped to {mapped_ticker})")
        for config in timeframe_configs:
            period = config['period']
            interval = config['interval']
            data = fetch_and_process_data([mapped_ticker], period, interval, False, 20, False, trade_log)
            if mapped_ticker in data:
                zones = identify_zones(data[mapped_ticker], interval, trade_log)
                trade_log.append(f"Found {len(zones)} zones for {ticker} at {period}/{interval}")
                for zone in zones:
                    zone['date'] = zone['date'].round('5min')
                    zone['period'] = period
                    zone['interval'] = interval
                    trade_log.append(f"Zone at {zone['level']:.2f} ({zone['type']}) on {zone['date']} for {period}/{interval}")
                all_zones.extend(zones)
            else:
                trade_log.append(f"No data for {mapped_ticker} at {period}/{interval}")
        
        # Step 1: Weekly + daily/hourly/minute super zones
        demand_zones = [z for z in all_zones if z['type'] == 'demand']
        supply_zones = [z for z in all_zones if z['type'] == 'supply']
        
        for zone_type in ['demand', 'supply']:
            zones = demand_zones if zone_type == 'demand' else supply_zones
            i = 0
            while i < len(zones):
                cluster = [zones[i]]
                j = i + 1
                while j < len(zones):
                    avg_level = np.mean([z['level'] for z in cluster])
                    threshold = 0.015 if '5m' in [z['interval'] for z in cluster] or '1m' in [z['interval'] for z in cluster] or '15m' in [z['interval'] for z in cluster] else 0.01
                    if abs(zones[j]['level'] - avg_level) <= avg_level * threshold:
                        cluster.append(zones[j])
                        zones.pop(j)
                    else:
                        j += 1
                intervals = set(z['interval'] for z in cluster)
                has_weekly = '1wk' in intervals
                has_daily_or_shorter = '1d' in intervals or '1h' in intervals or '30m' in intervals or '15m' in intervals or '5m' in intervals or '1m' in intervals
                if has_weekly and has_daily_or_shorter:
                    avg_level = np.mean([z['level'] for z in cluster])
                    super_zones.append({
                        'date': min(z['date'] for z in cluster),
                        'type': zone_type,
                        'level': avg_level,
                        'periods': list(set(z['period'] for z in cluster)),
                        'intervals': list(intervals)
                    })
                    trade_log.append(f"Super zone {zone_type} at {avg_level:.2f} with intervals {list(intervals)}")
                i += 1
        
        # Step 2: Intraday super zones (1mo/1h + 1mo/30m)
        intraday_zones = [z for z in all_zones if z['interval'] in ['1h', '30m'] and z['period'] == '1mo']
        intraday_demand = [z for z in intraday_zones if z['type'] == 'demand']
        intraday_supply = [z for z in intraday_zones if z['type'] == 'supply']
        
        for zone_type in ['demand', 'supply']:
            zones = intraday_demand if zone_type == 'demand' else intraday_supply
            i = 0
            while i < len(zones):
                cluster = [zones[i]]
                j = i + 1
                while j < len(zones):
                    avg_level = np.mean([z['level'] for z in cluster])
                    threshold = 0.01
                    if abs(zones[j]['level'] - avg_level) <= avg_level * threshold:
                        cluster.append(zones[j])
                        zones.pop(j)
                    else:
                        j += 1
                intervals = set(z['interval'] for z in cluster)
                if '1h' in intervals and '30m' in intervals:
                    avg_level = np.mean([z['level'] for z in cluster])
                    super_zones.append({
                        'date': min(z['date'] for z in cluster),
                        'type': zone_type,
                        'level': avg_level,
                        'periods': ['1mo'],
                        'intervals': list(intervals)
                    })
                    trade_log.append(f"Intraday super zone {zone_type} at {avg_level:.2f} with intervals {list(intervals)}")
                i += 1
        
        # Step 3: Intraday super zones (1d/5m + 1d/1m)
        intraday_zones_1d = [z for z in all_zones if z['interval'] in ['5m', '1m'] and z['period'] == '1d']
        intraday_demand_1d = [z for z in intraday_zones_1d if z['type'] == 'demand']
        intraday_supply_1d = [z for z in intraday_zones_1d if z['type'] == 'supply']
        
        for zone_type in ['demand', 'supply']:
            zones = intraday_demand_1d if zone_type == 'demand' else intraday_supply_1d
            i = 0
            while i < len(zones):
                cluster = [zones[i]]
                j = i + 1
                while j < len(zones):
                    avg_level = np.mean([z['level'] for z in cluster])
                    threshold = 0.015
                    if abs(zones[j]['level'] - avg_level) <= avg_level * threshold:
                        cluster.append(zones[j])
                        zones.pop(j)
                    else:
                        j += 1
                intervals = set(z['interval'] for z in cluster)
                if '5m' in intervals and '1m' in intervals:
                    avg_level = np.mean([z['level'] for z in cluster])
                    super_zones.append({
                        'date': min(z['date'] for z in cluster),
                        'type': zone_type,
                        'level': avg_level,
                        'periods': ['1d'],
                        'intervals': list(intervals)
                    })
                    trade_log.append(f"Intraday super zone {zone_type} at {avg_level:.2f} with intervals {list(intervals)}")
                i += 1
        
        trade_log.append(f"Found {len(super_zones)} super zones for {ticker}")
        return super_zones
    except Exception as e:
        trade_log.append(f"Error identifying super zones for {ticker}: {str(e)}")
        return []

def calculate_super_zone_probability(ticker, super_zones, trade_log):
    probabilities = []
    mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
    for sz in super_zones:
        zone_level = sz['level']
        zone_type = sz['type']
        data = fetch_and_process_data([mapped_ticker], 'max', '1d', True, 20, True, trade_log)
        if mapped_ticker not in data:
            continue
        df = data[mapped_ticker]
        instances = []
        future_data = df[df.index > sz['date']]
        if zone_type == 'demand':
            approaches = future_data[(future_data['low'] >= zone_level * 0.99) & (future_data['low'] <= zone_level * 1.01)]
        else:
            approaches = future_data[(future_data['high'] >= zone_level * 0.99) & (future_data['high'] <= zone_level * 1.01)]
        for approach_date in approaches.index:
            approach_price = df.loc[approach_date, 'close']
            post_approach = df[df.index > approach_date]
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
            instances.append({'outcome': outcome})
        held_count = sum(1 for inst in instances if inst['outcome'] == 'held')
        total = len(instances)
        prob_held = (held_count / total * 100) if total > 0 else 0
        probabilities.append({
            'level': zone_level,
            'type': zone_type,
            'probability_held': prob_held,
            'approaches': total
        })
    return probabilities

# Standalone function: Identify base and following candles
def identify_base_and_following_candles(data):
    base_rally_candles = []
    base_drop_candles = []
    for i in range(1, len(data) - 2):
        body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
        total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
        body_size_rally = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_rally = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        body_size_drop = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_drop = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        if (body_size_base <= 0.5 * total_height_base) and (body_size_rally >= 0.71 * total_height_rally) and (data['close'].iloc[i+1] > data['open'].iloc[i+1]):
            base_rally_candles.append(data.index[i])
        if (body_size_base <= 0.5 * total_height_base) and (body_size_drop >= 0.71 * total_height_drop) and (data['close'].iloc[i+1] < data['open'].iloc[i+1]):
            base_drop_candles.append(data.index[i])
    return base_rally_candles, base_drop_candles

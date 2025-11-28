import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os
import pytz
from datetime import datetime

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

# Cache for scanner data
@st.cache_data(ttl=900)
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
                try: score, approaches, age = scanner_calculate_zone_significance(df, zone, idx)
                except Exception: pass
                all_zones.append({'level': zone['level'], 'type': zone['type'], 'timeframe': tf, 'score': score,
                                  'approaches': approaches, 'age': age, 'date': zone['date'], 'tf_idx': idx})
    
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


def process_scan_results(scanner_tickers, all_scanner_data, periods_intervals, timeframes_list, tolerance, generate_trade_recommendation, identify_zones, build_relationship_chart):
    all_trades = []
    analysis_progress = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(scanner_tickers):
        status_text.text(f"Analyzing {ticker} ({i+1}/{len(scanner_tickers)})...")
        try:
            ticker_dfs = [None] * 4
            if ticker in all_scanner_data and all_scanner_data[ticker]:
                main_periods_list = list(periods_intervals.keys())
                main_timeframes_list = [item for sublist in periods_intervals.values() for item in sublist]
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


def scanner_tab_ui(periods_intervals, timeframes_list, tolerance, generate_trade_recommendation, identify_zones, build_relationship_chart):
    scanner_tab1, scanner_tab2, scanner_tab3 = st.tabs(["Scan", "Manage Watch Lists", "Results"])
    
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
                    
                    st.session_state.scanner_results = process_scan_results(scanner_tickers, all_scanner_data, periods_intervals, timeframes_list, tolerance, generate_trade_recommendation, identify_zones, build_relationship_chart)
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

                    st.session_state.scanner_results = process_scan_results(scanner_tickers, all_scanner_data, periods_intervals, timeframes_list, tolerance, generate_trade_recommendation, identify_zones, build_relationship_chart)
                    st.success("Scan complete! Check the 'Results' tab.")

    with scanner_tab2:
        st.header("Manage Watch Lists")
        market_watches_manage = load_market_watches()
        watch_list_name_manage = st.selectbox("Select Market Watch to Manage", options=list(market_watches_manage.keys()), key="manage_watchlist")

        st.subheader(f"Tickers in {watch_list_name_manage}")
        if market_watches_manage[watch_list_name_manage]:
            st.dataframe(pd.DataFrame(market_watches_manage[watch_list_name_manage], columns=["Ticker"] ), use_container_width=True)
        else:
            st.write("No tickers in this market watch.")
        
        st.divider() 
        
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

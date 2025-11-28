import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

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

def plot_trade_chart(df, zones, trades, symbol, timeframe, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, aligned_zones, show_aligned_zones, show_fresh_zones):
    if df is None or zones is None or not trades:
        st.session_state.trade_log.append(f"plot_trade_chart failed for {symbol} (Timeframe: {timeframe}, Period: {period})")
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
        if symbol in st.session_state.fresh_zones:
            for fz in st.session_state.fresh_zones[symbol]:
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

def backtesting_ui(final_ticker_list, timeframes_list, periods_list, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, show_aligned_zones, show_fresh_zones):
    backtest_button = st.button("Run Backtest")
    if 'backtest_data_ready' not in st.session_state:
        st.session_state.backtest_data_ready = False

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
                if 'trade_log' in st.session_state:
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

def trade_charts_ui(final_ticker_list, timeframes_list, periods_list, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, show_aligned_zones, show_fresh_zones):
    if st.session_state.get('backtest_data_ready', False) and any(any(trades for trades in tls.values()) for tls in st.session_state.trades_list.values()):
        for ticker in final_ticker_list:
            st.subheader(f"Trade Charts for {ticker}")
            cols = st.columns(2)
            for idx, (df, zones, trades, tf, period) in enumerate(zip(st.session_state.dfs[ticker], st.session_state.zones_list[ticker], st.session_state.trades_list[ticker], timeframes_list, periods_list)):
                if df is not None and zones is not None and trades:
                    with cols[idx % 2]:
                        fig = plot_trade_chart(df, zones, trades, ticker, tf, period, show_buy_zones, show_sell_zones, 
                                              show_limit_lines, show_prices, st.session_state.get('aligned_zones', {}), 
                                              show_aligned_zones, show_fresh_zones)
                        if fig:
                            st.pyplot(fig)
                            if 'trade_log' in st.session_state:
                                st.session_state.trade_log.append(f"Trade chart plotted for {ticker} (Timeframe {idx+1}: {tf})")
                            plt.close(fig)
    else:
        st.write("Run a backtest to view trade charts.")

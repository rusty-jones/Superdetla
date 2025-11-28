import yfinance as yf
import warnings
from config import ticker_mapping

warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Fetch and process data with EMA and RSI
def fetch_and_process_data(tickers, period, interval, use_ema, ema_period, use_rsi, trade_log):
    all_data = {}
    for ticker in tickers:
        try:
            mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
            if not mapped_ticker.startswith('^'):
                mapped_ticker = f"{ticker.upper()}.NS"
            trade_log.append(f"Fetching data for {ticker} ({mapped_ticker}, Period: {period}, Interval: {interval})")
            data = yf.download(mapped_ticker, period=period, interval=interval, progress=False)
            if data.empty:
                trade_log.append(f"No data returned for {ticker} ({mapped_ticker})")
                continue
            data.reset_index(inplace=True)
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            data.set_index('Date', inplace=True)
            data.index = data.index.tz_localize(None)
            trade_log.append(f"Fetched {len(data)} data points for {ticker}. First: {data.index[0]}, Last: {data.index[-1]}")
            if use_ema:
                data['ema'] = data['close'].ewm(span=ema_period, adjust=False).mean()
            if use_rsi:
                data['rsi'] = compute_rsi(data['close'], 14)
            data.dropna(inplace=True)
            trade_log.append(f"After processing, {len(data)} data points remain for {ticker}")
            all_data[ticker] = data
        except Exception as e:
            trade_log.append(f"Error fetching data for {ticker}: {str(e)}")
    return all_data

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

import streamlit as st
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import smtplib
from email.message import EmailMessage
from transformers import pipeline
import logging
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import white, black
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.platypus import TableStyle
import os
import json
import requests
import re
from valid_tickers import valid_tickers

# Import new modules
import scanner
import backtesting
import zone_analysis_gv

# Perplexity API configuration
API_KEY = "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"  # Replace with your valid Perplexity API key
API_URL = "https://api.perplexity.ai/chat/completions"

# --- Rest of the code from app-full.py ---
# (I will omit the code that is already in the file for brevity)
# ...

# Construct the prompt for Perplexity API
def create_prompt(json_data):
    prompt = """
You are an expert financial analyst. I have provided stock market data in JSON format, containing OHLC (Open, High, Low, Close, Volume) data, demand zones, and supply zones for multiple stocks across mid-term (3mo, 6mo, 1y) and long-term (1y, 2y, 5y) timeframes. Each stock entry includes:

- OHLC data: List of records with 'open', 'high', 'low', 'close', and 'volume' for each period.
- Demand zones: List of zones with 'date', 'low', 'high', and 'type' (demand), indicating potential buying areas.
- Supply zones: List of zones with 'date', 'low', 'high', and 'type' (supply), indicating potential selling areas.

For each stock, provide a concise and actionable analysis (30-50 words) based on the provided data. Include:
- A brief context of recent price action and key zones.
- A trading recommendation (buy, sell, or hold) with a clear rationale, focusing on price action relative to demand/supply zones and critical levels.
Formatted as:

**Stock: [Stock Name]**
Recommendation: [Recommendation (e.g., Buy/Sell/Hold)] - [Rationale, including key price levels, 30-50 words]

If the term "Recommendation" is not suitable, use a similar term like "Trading Recommendation" or "Advice". Keep the response concise and actionable. Here is the JSON data:

```json
{json_data}
```
"""
    return prompt.format(json_data=json.dumps(json_data, indent=2))

# Extract Recommendation section from the API response
def extract_recommendation(response_text):
    try:
        # Normalize response text (remove extra newlines, normalize spaces)
        response_text = re.sub(r'\n\s*\n', '\n\n', response_text.strip())
        # Split by stock sections
        stock_sections = re.split(r'\*\*Stock: (.*?)\*\*', response_text)[1:]
        recommendation_output = []
        for i in range(0, len(stock_sections), 2):
            stock_name = stock_sections[i].strip()
            content = stock_sections[i + 1].strip()
            # Flexible regex to match Recommendation, Trading Recommendation, Advice, or fallback to any text
            recommendation_match = re.search(
                r'(?:(?:Recommendation|Trading Recommendation|Advice)\s*[:\n]*\s*(.*?)(?=\n\n|\Z|$))|(.+?(?=\n\n|\Z|$))',
                content,
                re.DOTALL | re.IGNORECASE
            )
            if recommendation_match:
                # Use the first group if Recommendation/Trading Recommendation/Advice is found, else fallback to second group
                recommendation_text = (recommendation_match.group(1) or recommendation_match.group(2)).strip()
                recommendation_output.append(f"Stock: {stock_name}\nRecommendation: {recommendation_text}")
            else:
                recommendation_output.append(f"Stock: {stock_name}\nRecommendation: Not found in response.")
        return "\n\n".join(recommendation_output) if recommendation_output else "No recommendation sections found in response."
    except Exception as e:
        logger.error(f"Error extracting Recommendation: {e}")
        return f"Error extracting recommendation from response: {str(e)}"

# Send request to Perplexity API and return recommendation
def analyze_with_perplexity(json_data):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        prompt = create_prompt(json_data)
        payload = {
            "model": "sonar-pro",  # Updated to a supported model per Perplexity's latest docs
            "messages": [
                {"role": "system", "content": "You are a financial analyst providing concise stock market insights."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,  # Suitable for concise recommendations
            "temperature": 0.125
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_text = response.json()['choices'][0]['message']['content'].strip()
        logger.debug(f"Raw API response: {response_text}")
        return extract_recommendation(response_text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Perplexity API: {e}")
        return f"Failed to retrieve analysis from Perplexity API: {str(e)}"


chart_summaries = {}
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(initial_sidebar_state="collapsed")
st.cache_data.clear()

def debug_session_state():
    st.write("Debug: Session State Contents")
    st.write(st.session_state)

try:
    classifier = pipeline("text-classification", model="distilbert-base-uncased", device=-1)
except Exception as e:
    st.error(f"Error initializing Hugging Face classifier: {str(e)}")
    classifier = None

def validate_stocks(stocks_input):
    if not stocks_input or not isinstance(stocks_input, str):
        return [], []
    
    stock_list = [stock.strip().upper() for stock in stocks_input.split(",") if stock.strip()]
    valid_stocks = []
    invalid_stocks = []
    
    if classifier:
        for stock in stock_list:
            if stock in valid_tickers:
                valid_stocks.append(stock)
            else:
                try:
                    result = classifier(stock)
                    positive_score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score']
                    if positive_score > 0.7 and len(stock) <= 10 and stock.isupper():
                        valid_stocks.append(stock)
                    else:
                        invalid_stocks.append(stock)
                except Exception:
                    invalid_stocks.append(stock)
    else:
        for stock in stock_list:
            if stock in valid_tickers or (len(stock) <= 10 and stock.isupper() and stock.isalnum()):
                valid_stocks.append(stock)
            else:
                invalid_stocks.append(stock)
    
    logger.debug(f"Validated stocks: {valid_stocks}, Invalid stocks: {invalid_stocks}")
    return valid_stocks, invalid_stocks

# Global definitions for periods and intervals
short_term_periods_intervals = {"1d": ["5m"], "5d": ["30m"], "1mo": ["1h", "1d"],"3mo": ["4h"]}
mid_term_periods_intervals = {"3mo": ["1d"], "6mo": ["1wk"], "1y": ["1d"]}
long_term_periods_intervals = {"1y": ["1wk"], "2y": ["1mo"], "5y": ["1mo"]}

def validate_period_interval(periods, intervals):
    valid_combinations = []
    invalid_combinations = []
    valid_intervals_for_period = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
        "1mo": ["30m", "60m", "90m", "1h", "1d"],
        "3mo": ["1h", "1d", "5d", "1wk"],
        "6mo": ["1h", "1d", "5d", "1wk"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo", "3mo"],
        "10y": ["1d", "5d", "1wk", "1mo", "3mo"],
        "ytd": ["1h", "1d", "5d", "1wk"],
        "max": ["1d", "5d", "1wk", "1mo", "3mo"]
    }
    for period in periods:
        for interval in intervals:
            if period in valid_intervals_for_period and interval in valid_intervals_for_period[period]:
                valid_combinations.append((period, interval))
            else:
                invalid_combinations.append((period, interval))
    logger.debug(f"Valid period-interval combinations: {valid_combinations}, Invalid: {invalid_combinations}")
    return valid_combinations, invalid_combinations

for key in list(st.session_state.keys()):
    if key not in ["filter_option", "filter_selected_intervals", "radio_filter", "selected_stocks", "stocks_input", "checkbox_states", "analysis_results", "custom_periods", "custom_intervals", "clean_results"]:
        del st.session_state[key]

if "filter_option" not in st.session_state:
    st.session_state.filter_option = "Monthly Only"
if "filter_selected_intervals" not in st.session_state:
    st.session_state.filter_selected_intervals = ["1mo"]
if "radio_filter" not in st.session_state:
    st.session_state.radio_filter = "Monthly Only"
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "stocks_input" not in st.session_state:
    st.session_state.stocks_input = ""
if "checkbox_states" not in st.session_state:
    st.session_state.checkbox_states = {}
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "custom_periods" not in st.session_state:
    st.session_state.custom_periods = []
if "custom_intervals" not in st.session_state:
    st.session_state.custom_intervals = []
if "clean_results" not in st.session_state:
    st.session_state.clean_results = {}

def update_custom_periods():
    logger.debug(f"Updating custom_periods: {st.session_state.custom_periods}")

def update_custom_intervals():
    logger.debug(f"Updating custom_intervals: {st.session_state.custom_intervals}")

stock_lists = {
    "NSE FNO": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "NSE Midcap": ["ADANIPOWER", "BANKBARODA", "GAIL", "HINDPETRO"],
    "NSE Smallcap": ["BSE", "CDSL", "IRB", "MAHAPEX"]
}

checkbox_stock_lists = {
    "FNO": ["360ONE","AARTIIND","ABB","ABCAPITAL","ABFRL","ACC","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ALKEM","AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","ASTRAL","ATGL","AUBANK",
"AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BALKRISIND","BANDHANBNK","BANKBARODA","BANKINDIA","BDL","BEL","BHARATFORG","BHARTIARTL","BHEL","BIOCON","BLUESTARCO","BOSCHLTD","BPCL","BRITANNIA",
"BSE","BSOFT","CAMS","CANBK","CDSL","CESC","CGPOWER","CHAMBLFERT","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND","CYIENT","DABUR","DALBHARAT","DELHIVERY","DIVISLAB","DIXON","DLF","DMART",
"DRREDDY","EICHERMOT","ETERNAL","EXIDEIND","FEDERALBNK","FORTIS","GAIL","GLENMARK","GMRAIRPORT","GODREJCP","GODREJPROP","GRANULES","GRASIM","HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO","HFCL",
"HINDALCO","HINDCOPPER","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","ICICIBANK","ICICIGI","ICICIPRULI","IDEA","IDFCFIRSTB","IEX","IGL","IIFL","INDHOTEL","INDIANB","INDIGO","INDUSINDBK","INDUSTOWER","INFY","INOXWIND",
"IOC","IRB","IRCTC","IREDA","IRFC","ITC","JINDALSTEL","JIOFIN","JSL","JSWENERGY","JSWSTEEL","JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KOTAKBANK","KPITTECH","LAURUSLAB","LICHSGFIN","LICI","LODHA","LT","LTF","LTIM",
"LUPIN","M&M","M&MFIN","MANAPPURAM","MANKIND","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL","MGL","MOTHERSON","MPHASIS","MUTHOOTFIN","NATIONALUM","NAUKRI","NBCC","NCC","NESTLEIND","NHPC","NMDC","NTPC","NYKAA",
"OBEROIRLTY","OFSS","OIL","ONGC","PAGEIND","PATANJALI","PAYTM","PEL","PERSISTENT","PETRONET","PFC","PGEL","PHOENIXLTD","PIDILITIND","PIIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB","POONAWALLA","POWERGRID","PPLPHARMA",
"PRESTIGE","RBLBANK","RECLTD","RELIANCE","RVNL","SAIL","SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SJVN","SOLARINDS","SONACOMS","SRF","SUNPHARMA","SUPREMEIND","SYNGENE","TATACHEM","TATACOMM","TATACONSUM",
"TATAELXSI","TATAMOTORS","TATAPOWER","TATASTEEL","TATATECH","TCS","TECHM","TIINDIA","TITAGARH","TITAN","TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNIONBANK","UNITDSPR","UNOMINDA","UPL","VBL","VEDL","VOLTAS","WIPRO","YESBANK","ZYDUSLIFE"]
}

for list_name, stocks in checkbox_stock_lists.items():
    if list_name not in st.session_state.checkbox_states:
        st.session_state.checkbox_states[list_name] = {stock: False for stock in stocks}

stocks_input = st.session_state.get("stocks_input", "")
valid_stocks, invalid_stocks = validate_stocks(stocks_input)

def fetch_and_process_data(stocks, periods_intervals):
    all_data = {}
    for stock in stocks:
        all_data[stock] = {}
        for period, intervals in periods_intervals.items():
            for interval in intervals:
                try:
                    logger.debug(f"Fetching data for {stock}, period: {period}, interval: {interval}")
                    data = yf.download(stock, period=period, interval=interval)
                    if not data.empty:
                        data.reset_index(inplace=True)
                        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                        data.set_index('Date', inplace=True)
                        all_data[stock][(period, interval)] = data
                    else:
                        logger.warning(f"No data fetched for {stock}, period: {period}, interval: {interval}")
                except Exception as e:
                    logger.error(f"Error fetching data for {stock}, period: {period}, interval: {interval}: {str(e)}")
    return all_data

def identify_base_and_following_candles(data, max_zones=5, max_overlap_pct=30, require_last_in_zone=True):
    base_rally_candles = []
    base_drop_candles = []
    for i in range(1, len(data) - 2):
        body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
        total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
        body_size_follow = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_follow = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        if total_height_base > 0 and total_height_follow > 0:
            if (body_size_base <= 0.5 * total_height_base) and (body_size_follow >= 0.71 * total_height_follow) and (data['close'].iloc[i+1] > data['open'].iloc[i+1]):
                base_rally_candles.append({'date': data.index[i], 'low': float(data['low'].iloc[i]), 'high': float(max(data['open'].iloc[i], data['close'].iloc[i])), 'type': 'demand'})
            if (body_size_base <= 0.5 * total_height_base) and (body_size_follow >= 0.71 * total_height_follow) and (data['close'].iloc[i+1] < data['open'].iloc[i+1]):
                base_drop_candles.append({'date': data.index[i], 'low': float(min(data['open'].iloc[i], data['close'].iloc[i])), 'high': float(data['high'].iloc[i]), 'type': 'supply'})
    
    all_zones = base_rally_candles + base_drop_candles
    is_clean = len(all_zones) <= max_zones
    overlap_count = 0
    if len(all_zones) > 1 and all_zones:
        zone_spread = max(z['high'] for z in all_zones) - min(z['low'] for z in all_zones)
        if zone_spread > 0:
            overlap_count = sum(1 for i in range(len(all_zones)) for j in range(i + 1, len(all_zones))
                            if abs(all_zones[i]['high'] - all_zones[j]['low']) / zone_spread <= 0.01 or abs(all_zones[j]['high'] - all_zones[i]['low']) / zone_spread <= 0.01)
            is_clean = is_clean and (overlap_count / len(all_zones) * 100 < max_overlap_pct)
    
    is_last_in_zone = False
    if all_zones and len(data) >= 2:
        last_two_candles = data.iloc[-2:][['open', 'high', 'low', 'close']].values.flatten()
        is_last_in_zone = any(zone['low'] <= price <= zone['high'] for zone in all_zones for price in last_two_candles)
    
    if require_last_in_zone:
        is_clean = is_clean and is_last_in_zone
    
    return base_rally_candles, base_drop_candles, is_clean

def generate_json_data(stocks):
    json_data = {}
    periods_intervals = {**mid_term_periods_intervals, **long_term_periods_intervals}
    all_data = fetch_and_process_data(stocks, periods_intervals)
    
    for stock in stocks:
        json_data[stock] = {}
        for (period, interval), data in all_data.get(stock, {}).items():
            if not data.empty:
                base_rally_candles, base_drop_candles, _ = identify_base_and_following_candles(data, max_zones=5, max_overlap_pct=30, require_last_in_zone=True)
                # Convert Timestamps to strings in OHLC data
                ohlc = data[['open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
                # Convert Timestamps to strings in zones
                for zone in base_rally_candles + base_drop_candles:
                    if isinstance(zone['date'], pd.Timestamp):
                        zone['date'] = zone['date'].strftime('%Y-%m-%d')
                json_data[stock][f"{period}_{interval}"] = {
                    'ohlc': ohlc,
                    'demand_zones': base_rally_candles,
                    'supply_zones': base_drop_candles
                }
    
    return json_data

def plot_candlestick_chart(df, base_rally_candles, base_drop_candles, title, style='nightclouds', save_path=None):
    if df.empty:
        logger.warning(f"No data to plot for {title}")
        return
    try:
        fig, ax = mpf.plot(df, type='candle', style=style, title=title, ylabel='Price', volume=False, returnfig=True)
        for zone in base_rally_candles:
            date = zone['date']
            if date in df.index:
                idx = df.index.get_loc(date)
                low_candle = zone['low']
                high_body = zone['high']
                if high_body > low_candle:
                    rect = patches.Rectangle((idx - 0.4, low_candle), len(df) - idx + 0.4, high_body - low_candle,
                                            linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
                    ax[0].add_patch(rect)
        for zone in base_drop_candles:
            date = zone['date']
            if date in df.index:
                idx = df.index.get_loc(date)
                low_body = zone['low']
                high_candle = zone['high']
                if high_candle > low_body:
                    rect = patches.Rectangle((idx - 0.4, low_body), len(df) - idx + 0.4, high_candle - low_body,
                                            linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
                    ax[0].add_patch(rect)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        logger.error(f"Error plotting chart {title}: {str(e)}")
        st.error(f"Error plotting chart: {str(e)}")

def plot_quick_analysis(df, base_rally_candles, base_drop_candles, title):
    if not base_rally_candles and not base_drop_candles:
        logger.warning(f"No zones to plot for quick analysis {title}")
        return
    try:
        fig, ax = mpf.plot(df, type='candle', style='nightclouds', title=title, ylabel='Price', volume=False, returnfig=True)
        for zone in base_rally_candles:
            date = zone['date']
            if date in df.index:
                idx = df.index.get_loc(date)
                low_candle = zone['low']
                high_body = zone['high']
                if high_body > low_candle:
                    rect = patches.Rectangle((idx - 0.4, low_candle), len(df) - idx + 0.4, high_body - low_candle,
                                            linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
                    ax[0].add_patch(rect)
                    ax[0].text(len(df) - 1, high_body, f'{high_body:.2f}', color='blue', va='center', ha='left')
        for zone in base_drop_candles:
            date = zone['date']
            if date in df.index:
                idx = df.index.get_loc(date)
                low_body = zone['low']
                high_candle = zone['high']
                if high_candle > low_body:
                    rect = patches.Rectangle((idx - 0.4, low_body), len(df) - idx + 0.4, high_candle - low_body,
                                            linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
                    ax[0].add_patch(rect)
                    ax[0].text(len(df) - 1, low_body, f'{low_body:.2f}', color='red', va='center', ha='left')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error plotting quick analysis chart {title}: {str(e)}")
        st.error(f"Error plotting quick analysis chart: {str(e)}")

def generate_stock_analysis(stock, data_dict):
    global chart_summaries
    analysis_text = f"Analysis for {stock.replace('.NS', '')}:\n"
    summaries = chart_summaries.get(stock, {})
    demand_zones, supply_zones = [], []
    last_candle_in_zone = None
    
    # Use generate_json_data for Perplexity API
    json_data = generate_json_data([stock])
    perplexity_analysis = analyze_with_perplexity(json_data) if json_data.get(stock) else "No data available for Perplexity analysis."
    
    for key, data in data_dict.items():
        if '_' in key:
            period, interval = key.split('_')
            base_rally_candles, base_drop_candles, is_clean = identify_base_and_following_candles(data, max_zones=5, max_overlap_pct=30, require_last_in_zone=True)
            demand_zones.extend(base_rally_candles)
            supply_zones.extend(base_drop_candles)
            
            if key == "3mo_1d" and not data.empty:
                last_candle = data.iloc[-1]
                last_price_range = [last_candle['open'], last_candle['close'], last_candle['high'], last_candle['low']]
                for zone in base_rally_candles:
                    if any(zone['low'] <= price <= zone['high'] for price in last_price_range):
                        last_candle_in_zone = 'demand'
                        break
                for zone in base_drop_candles:
                    if any(zone['low'] <= price <= zone['high'] for price in last_price_range):
                        last_candle_in_zone = 'supply'
                        break
            
            if is_clean and (base_rally_candles or base_drop_candles):
                zone_summary = f"{period} ({interval}): {len(base_rally_candles)} demand zone(s), {len(base_drop_candles)} supply zone(s)"
                if last_candle_in_zone and key == "3mo_1d":
                    zone_summary += f", last candle in {last_candle_in_zone} zone"
                analysis_text += f"- {zone_summary}\n"
                summaries[key] = zone_summary
            else:
                analysis_text += f"- {period} ({interval}): No clean zones identified\n"
                summaries[key] = f"No clean zones in {period} ({interval})"
    
    chart_summaries[stock] = summaries
    decision = "Neutral"
    if demand_zones or supply_zones:
        demand_count = len(demand_zones)
        supply_count = len(supply_zones)
        if last_candle_in_zone == 'demand' and demand_count > supply_count:
            decision = "Bullish"
            analysis_text += f"\nDecision: Bullish signal detected. Last candle in demand zone on 3mo daily chart, with {demand_count} demand zones vs {supply_count} supply zones."
        elif last_candle_in_zone == 'supply' and supply_count > demand_count:
            decision = "Bearish"
            analysis_text += f"\nDecision: Bearish signal detected. Last candle in supply zone on 3mo daily chart, with {supply_count} supply zones vs {demand_count} demand zones."
        else:
            analysis_text += f"\nDecision: Neutral. No clear bullish or bearish signal based on {demand_count} demand zones and {supply_count} supply zones."
    
    # Combine existing analysis with Perplexity analysis
    combined_analysis = f"{analysis_text}\n Analysis:\n{perplexity_analysis}"
    
    return {"analysis": combined_analysis.strip(), "decision": decision}

def generate_summary_table(clean_results, selected_intervals, return_data_only=False):
    try:
        data = []
        for stock, details in clean_results.items():
            intervals = sorted(set(details['intervals']))
            periods = sorted(set(details['periods']))
            if intervals:
                data.append({
                    'Stock': stock.replace('.NS', ''),
                    'Intervals': ', '.join(intervals),
                    'Periods': ', '.join(periods)
                })
        if data:
            df = pd.DataFrame(data)
            if return_data_only:
                return df  # Return DataFrame for PDF generation
            else:
                st.markdown("<h3 style='color: #66FFCC; text-align: center;'></h3>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
        else:
            if not return_data_only:
                st.markdown("<p style='color: #FF4D4D; text-align: center;'>No clean charts found for the selected intervals.</p>", unsafe_allow_html=True)
            return pd.DataFrame()  # Return empty DataFrame for PDF generation
    except Exception as e:
        logger.error(f"Error generating summary table: {str(e)}")
        if not return_data_only:
            st.error(f"Error generating summary table: {str(e)}", icon="🚨")
        return pd.DataFrame()

def generate_all_charts_pdf(stocks, output_path="all_charts_stock_analysis.pdf"):
    logger.debug(f"Generating PDF for all charts with stocks: {stocks}")
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Set black background for the entire PDF
    c.setFillColorRGB(0, 0, 0)  # Black background
    c.rect(0, 0, width, height, fill=1)
    
    y_position = 700
    c.setFont("Helvetica", 14)
    c.setFillColor(white)  # White text for visibility
    c.drawString(50, y_position, "All Charts Stock Analysis Report")
    y_position -= 30

    # Define all periods and intervals
    all_periods_intervals = {
        **intraday_periods_intervals,
        **short_term_periods_intervals,
        **long_term_periods_intervals
    }
    
    # Fetch data for all stocks
    all_data = fetch_and_process_data(stocks, all_periods_intervals)
    
    # Generate summary table data
    summary_data = []
    for stock in stocks:
        display_stock = stock.replace(".NS", "")
        periods = []
        intervals = []
        for (period, interval), data in all_data.get(stock, {}).items():
            if not data.empty:
                periods.append(period)
                intervals.append(interval)
        if periods and intervals:
            summary_data.append({
                'Stock': display_stock,
                'Periods': ', '.join(sorted(set(periods))),
                'Intervals': ', '.join(sorted(set(intervals)))
            })

    # Add summary table
    if summary_data:
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, "All Charts Summary")
        y_position -= 20
        c.setFont("Helvetica", 10)
        
        headers = ["Stock", "Periods", "Intervals"]
        col_widths = [150, 200, 200]
        row_height = 20
        c.setFillColorRGB(0.2, 0.2, 0.2)  # Dark gray for table header
        c.rect(50, y_position - row_height, sum(col_widths), row_height, fill=1)
        c.setFillColor(white)
        c.drawString(50, y_position - 15, headers[0])
        c.drawString(50 + col_widths[0], y_position - 15, headers[1])
        c.drawString(50 + col_widths[0] + col_widths[1], y_position - 15, headers[2])
        y_position -= row_height
        
        for idx, row in enumerate(summary_data):
            if idx % 2 == 0:
                c.setFillColorRGB(0.15, 0.15, 0.15)  # Alternating row colors
            else:
                c.setFillColorRGB(0.1, 0.1, 0.1)
            c.rect(50, y_position - row_height, sum(col_widths), row_height, fill=1)
            c.setFillColor(white)
            c.drawString(50, y_position - 15, row['Stock'])
            c.drawString(50 + col_widths[0], y_position - 15, row['Periods'])
            c.drawString(50 + col_widths[0] + col_widths[1], y_position - 15, row['Intervals'])
            y_position -= row_height
            if y_position < 100:
                c.showPage()
                c.setFillColorRGB(0, 0, 0)
                c.rect(0, 0, width, height, fill=1)
                y_position = 700
                c.setFillColor(white)
                c.setFont("Helvetica", 12)
                c.drawString(50, y_position, "All Charts Summary (Continued)")
                y_position -= 20
                c.setFont("Helvetica", 10)
                c.setFillColorRGB(0.2, 0.2, 0.2)
                c.rect(50, y_position - row_height, sum(col_widths), row_height, fill=1)
                c.setFillColor(white)
                c.drawString(50, y_position - 15, headers[0])
                c.drawString(50 + col_widths[0], y_position - 15, headers[1])
                c.drawString(50 + col_widths[0] + col_widths[1], y_position - 15, headers[2])
                y_position -= row_height
        y_position -= 20
        c.showPage()
        c.setFillColorRGB(0, 0, 0)
        c.rect(0, 0, width, height, fill=1)
        y_position = 700
    else:
        c.setFont("Helvetica", 10)
        c.setFillColor(white)
        c.drawString(50, y_position, "No charts found for selected stocks.")
        y_position -= 20
        c.showPage()
        c.setFillColorRGB(0, 0, 0)
        c.rect(0, 0, width, height, fill=1)
        y_position = 700

    # Add one chart per page for each stock
    chart_paths = []
    for stock in stocks:
        display_stock = stock.replace(".NS", "")
        for (period, interval), data in all_data.get(stock, {}).items():
            if not data.empty:
                c.setFillColorRGB(0, 0, 0)  # Reset black background for new page
                c.rect(0, 0, width, height, fill=1)
                y_position = 700
                base_rally_candles, base_drop_candles, _ = identify_base_and_following_candles(data)
                chart_path = f"charts/{stock}_{period}_{interval}_all.png"
                os.makedirs("charts", exist_ok=True)
                plot_candlestick_chart(
                    data,
                    base_rally_candles,
                    base_drop_candles,
                    f"{display_stock} - {period} {interval}",
                    style='nightclouds',
                    save_path=chart_path
                )
                if os.path.exists(chart_path):
                    chart_paths.append(chart_path)
                    c.setFont("Helvetica", 12)
                    c.setFillColor(white)
                    c.drawString(50, y_position, f"{display_stock} - {period} {interval}")
                    y_position -= 20
                    c.drawImage(chart_path, 50, y_position - 400, width=500, height=400)
                    c.showPage()  # New page for each chart
    
    c.save()
    
    # Clean up chart files
    for chart_path in chart_paths:
        if os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception as e:
                logger.warning(f"Error removing chart file {chart_path}: {str(e)}")
    
    logger.debug(f"PDF generated at {output_path}")
    return output_path

def generate_pdf(clean_results, stock_list, max_zones, max_overlap_pct, require_last_in_zone, output_path="stock_analysis.pdf"):
    try:
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        # Set black background for the first page
        c.setFillColor(colors.black)
        c.rect(0, 0, width, height, fill=1)
        c.setFont("Helvetica", 16)
        c.setFillColor(colors.white)  # White text for visibility
        c.drawCentredString(width / 2, height - 40, "Stock Analysis Report")
        
        # Generate summary table data
        df = generate_summary_table(clean_results, st.session_state.filter_selected_intervals, return_data_only=True)
        
        # Draw summary table with dark theme matching all chart format
        if not df.empty:
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.white)
            c.drawString(50, height - 80, "")
            table_data = [df.columns.tolist()] + df.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.black),         # Black header background
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.lightgrey),     # Light grey text for header
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.black),       # Black row background
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.lightgrey),    # Light grey text for rows
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),      # Light grey grid lines
                ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey)        # Light grey border
            ]))
            table.wrapOn(c, width - 100, height - 150)
            table.drawOn(c, 50, height - 150 - len(table_data) * 20)
            c.showPage()
        else:
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.white)
            c.drawString(50, height - 80, "No clean charts available for the selected intervals.")
            c.showPage()

        # Draw charts without headings
        for stock in stock_list:
            if stock in clean_results:
                for chart_path in clean_results[stock].get('chart_paths', []):
                    if os.path.exists(chart_path):
                        # Set black background for chart page
                        c.setFillColor(colors.black)
                        c.rect(0, 0, width, height, fill=1)
                        c.drawImage(chart_path, 50, height - 450, width=width - 100, height=400, preserveAspectRatio=True)
                        c.showPage()

        c.save()
        return output_path
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise

def embed_tradingview_chart(symbol):
    try:
        tv_symbol = symbol.replace(".NS", "") if symbol.endswith(".NS") else symbol
        tv_symbol_map = {"^NSEI": "NSE:NIFTY", "^NSEBANK": "NSE:BANKNIFTY", "^BSESN": "BSE:SENSEX"}
        tv_symbol = tv_symbol_map.get(symbol, f"NSE:{tv_symbol}")
        tradingview_html = f"""
        <div class="tradingview-widget-container" style="height: 200px; width: 100%; margin-bottom: 20px; overflow-x: auto;">
          <div id="tradingview_{symbol}" style="height: 100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "D",
            "timezone": "Asia/Kolkata",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#1A1D26",
            "enable_publishing": false,
            "hide_top_toolbar": true,
            "hide_legend": true,
            "save_image": false,
            "container_id": "tradingview_{symbol}"
          }}
          );
          </script>
        </div>
        """
        st.markdown(tradingview_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error embedding TradingView chart for {symbol}: {str(e)}")
        st.error(f"Failed to load TradingView chart for {symbol}.", icon="🚨")

def send_email(name, email, message):
    sender_email = "your_email@gmail.com"  # Replace with your email
    sender_password = "your_app_password"  # Replace with your app-specific password
    receiver_email = "your_email@gmail.com"  # Replace with your email
    msg = EmailMessage()
    msg.set_content(f"Name: {name}\nEmail: {email}\nMessage: {message}")
    msg["Subject"] = "New Contact Form Submission"
    msg["From"] = sender_email
    msg["To"] = receiver_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception:
        return False

def sync_stock_input():
    selected = []
    for list_name, stocks in checkbox_stock_lists.items():
        select_all_key = f"select_all_{list_name}"
        if st.session_state.get(select_all_key, False):
            selected.extend(stocks)
            st.session_state.checkbox_states[list_name] = {stock: True for stock in stocks}
        else:
            for stock in stocks:
                checkbox_key = f"stock_{list_name}_{stock}"
                if st.session_state.get(checkbox_key, False):
                    selected.append(stock)
                    st.session_state.checkbox_states[list_name][stock] = True
                else:
                    st.session_state.checkbox_states[list_name][stock] = False
    st.session_state.selected_stocks = list(set(selected))
    logger.debug(f"Synced selected stocks: {st.session_state.selected_stocks}")

def clear_stocks():
    st.session_state.selected_stocks = []
    st.session_state.checkbox_states = {list_name: {stock: False for stock in stocks} for list_name, stocks in checkbox_stock_lists.items()}
    st.session_state.analysis_results = {}
    st.session_state.clean_results = {}
    for list_name in checkbox_stock_lists:
        select_all_key = f"select_all_{list_name}"
        if st.session_state.get(select_all_key, False):
            st.session_state[select_all_key] = False
    logger.debug("Cleared selected stocks and session state")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Home"):
        for key in list(st.session_state.keys()):
            if key not in ["filter_option", "filter_selected_intervals", "radio_filter", "selected_stocks", "stocks_input", "checkbox_states", "analysis_results", "custom_periods", "custom_intervals", "clean_results"]:
                del st.session_state[key]
        st.session_state.show_contact_form = False
        st.session_state.analysis_results = {}
        st.session_state.clean_results = {}
        st.rerun()
with col2:
    if st.button("Contact"):
        for key in list(st.session_state.keys()):
            if key not in ["filter_option", "filter_selected_intervals", "radio_filter", "selected_stocks", "stocks_input", "checkbox_states", "custom_periods", "custom_intervals", "clean_results"]:
                del st.session_state[key]
        st.session_state.show_contact_form = True
        st.session_state.analysis_results = {}
        st.session_state.clean_results = {}
        st.rerun()

if st.sidebar.button("Reset Session State"):
    st.session_state.clear()
    st.rerun()

st.sidebar.header("Criteria")
analysis_method = st.sidebar.selectbox("Analysis Method", ["Base/Following Candles", "GV Zone Analysis"])
max_zones = st.sidebar.number_input("Max Zones", min_value=1, max_value=20, value=5, step=1)
max_overlap_pct = st.sidebar.number_input("Max Overlap %", min_value=0, max_value=100, value=30, step=1)
require_last_in_zone = st.sidebar.checkbox("Require Last Candle in Zone", value=True)

filter_options_map = {
    "Daily": ["1d"],
    "Weekly Only": ["1wk"],
    "Monthly Only": ["1mo"],
    "Daily and Weekly": ["1d", "1wk"],
    "Daily, Weekly, and Monthly": ["1d", "1wk", "1mo"]
}

def update_filter_intervals():
    if st.session_state.radio_filter in filter_options_map:
        st.session_state.filter_option = st.session_state.radio_filter
        st.session_state.filter_selected_intervals = filter_options_map[st.session_state.radio_filter]
    logger.debug(f"Updated filter intervals: {st.session_state.filter_selected_intervals}")

st.sidebar.header("Interval Filters")
with st.sidebar:
    with st.form("filter_form"):
        filter_selection = st.radio(
            "Select Interval",
            options=["Daily", "Weekly Only", "Monthly Only", "Daily and Weekly", "Daily, Weekly, and Monthly"],
            key="radio_filter",
            index=["Daily", "Weekly Only", "Monthly Only", "Daily and Weekly", "Daily, Weekly, and Monthly"].index(st.session_state.filter_option) if st.session_state.filter_option in filter_options_map else 2
        )
        submit_button = st.form_submit_button("Apply Filter", on_click=update_filter_intervals)

intraday_periods_intervals = {"1d": ["5m"], "5d": ["30m"], "1mo": ["1h", "1d"]}
short_term_periods_intervals = {"1d": ["5m"], "5d": ["30m"], "1mo": ["1h", "1d"],"3mo": ["4h"]}
long_term_periods_intervals = {"1y": ["1wk"], "2y": ["1mo"], "5y": ["1mo"]}
chart_labels = {
    "1d_5m": "Short-Term Chart", "5d_30m": "Short-Term Chart", "1mo_1h": "Short-Term Chart", "1mo_1d": "Short-Term Chart",
    "3mo_1d": "Mid-Term Chart", "6mo_1wk": "Mid-Term Chart", "1y_1d": "Mid-Term Chart",
    "1y_1wk": "Long-Term Chart", "2y_1mo": "Long-Term Chart", "5y_1mo": "Long-Term Chart"
}
quick_periods_intervals = {"1d": ["5m"], "5d": ["30m"]}
crypto_periods_intervals = {"1d": ["5m", "15m", "30m", "1h"]}

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    .stTabs [role="tab"] { font-size: 20px; color: #FAFAFA; background-color: #1A1D26; border-radius: 8px 8px 0 0; padding: 10px 20px; margin-right: 4px; transition: background-color 0.3s ease; }
    .stTabs [role="tab"][aria-selected="true"] { background-color: #66FFCC; color: #000000; font-weight: 600; }
    .stTabs [role="tab"]:hover { background-color: #2A2E3A; }
    .stTextInput > div > div > input { width: 100% !important; max-width: 800px; background-color: #1A1D26; color: #FAFAFA; border: 1px solid #3A3F4A; border-radius: 8px; padding: 10px; font-size: 16px; transition: border-color 0.3s ease; }
    .stTextInput > div > div > input:focus { border-color: #66FFCC; box-shadow: 0 0 5px rgba(0, 255, 153, 0.3); }
    .stButton > button { background-color: #000000; color: #FAFAFA; border: 1px solid #66FFCC; border-radius: 8px; padding: 8px 16px; font-size: 14px; font-weight: 600; transition: background-color 0.3s ease, transform 0.2s ease; }
    .stButton > button:hover { background-color: #00FF00; color: #000000; transform: translateY(-2px); }
    .stButton > button:active { background-color: #00CC00; color: #000000; transform: translateY(0); }
    .checkbox-label { font-size: 14px !important; font-weight: normal !important; color: #D3D3D3 !important; }
    .stCheckbox > label { color: #D3D3D3; font-size: 14px; }
    .streamlit-expanderHeader { background-color: #1A1D26; color: #FAFAFA; border-radius: 8px; padding: 10px; font-size: 16px; font-weight: 500; }
    .streamlit-expander { border: 1px solid #3A3F4A; border-radius: 8px; background-color: #151821; }
    [data-testid="stSidebar"] { background-color: #151821; border-right: 1px solid #3A3F4A; }
    h1, h2, h3, h4, h5 { color: #FAFAFA; font-weight: 600; }
    p { color: #D3D3D3; font-size: 14px; }
    .stock-display { border: 2px solid #66FFCC; border-radius: 8px; padding: 12px; background-color: #1A1D26; margin-top: 10px; }
    .summary-box { border: 2px solid #66FFCC; border-radius: 8px; padding: 12px; background-color: #1A1D26; margin-top: 10px; }
    hr { border: 1px solid #3A3F4A; margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

summary_tab, charts_tab, scanner_tab, backtesting_tab, trade_charts_tab, testing_tab = st.tabs(["Search", "Charts", "Scanner", "Backtesting", "Trade Charts", "Testing"])


with summary_tab:
    st.markdown("<h1 style='text-align: center; font-size: 28px; margin-bottom: 20px;'>Stock Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    try:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.empty()
        with col2:
            stocks_input = st.text_input("Enter Stocks (e.g., TCS, INFY)", value=st.session_state.get("stocks_input", ""), key="stocks_input", placeholder="Enter stock symbols separated by commas")
            
            with st.expander("Or Add Stocks from List", expanded=False):
                try:
                    with st.form("stock_selection_form"):
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            add_button = st.form_submit_button("Add Stocks", on_click=sync_stock_input)
                        with col_btn2:
                            clear_button = st.form_submit_button("Clear Stocks", on_click=clear_stocks)
                        
                        for list_name, stocks in checkbox_stock_lists.items():
                            st.markdown(f"<h4 style='color: #66FFCC; font-size: 16px; font-weight: normal;'>{list_name}</h4>", unsafe_allow_html=True)
                            select_all_key = f"select_all_{list_name}"
                            st.checkbox(f"Select All ({list_name})", value=st.session_state.get(select_all_key, False), key=select_all_key)
                            cols = st.columns(4)
                            for idx, stock in enumerate(stocks):
                                with cols[idx % 4]:
                                    checkbox_key = f"stock_{list_name}_{stock}"
                                    st.checkbox(stock, value=st.session_state.checkbox_states[list_name].get(stock, False), key=checkbox_key, label_visibility="visible")
                
                except Exception as e:
                    st.error(f"Error rendering stock selection form: {str(e)}", icon="🚨")
                    debug_session_state()
            
            col_cb, col_btn = st.columns([1, 1])
            with col_cb:
                append_ns = st.checkbox("NSE stocks", value=True)
            with col_btn:
                search_trades_button = st.button("Show Analysis")
            
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            with col_btn1:
                short_term_button = st.button("Short Term")
            with col_btn2:
                mid_term_button = st.button("Mid-Term")
            with col_btn3:
                long_term_button = st.button("Long-Term")
            with col_btn4:
                index_analysis_button = st.button("Index Analysis")
            
            st.markdown("<hr>", unsafe_allow_html=True)

            all_stocks = list(set(valid_stocks + st.session_state.selected_stocks))
            if all_stocks:
                stock_display = ", ".join([stock.replace(".NS", "") for stock in all_stocks])  # Remove .NS for display
                st.markdown(
                    f"""
                    <div class='stock-display'>
                        <p style='font-size: 16px; color: #FAFAFA;'>{stock_display}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class='stock-display'>
                        <p style='font-size: 16px; color: #FAFAFA;'>No valid stocks selected or entered.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        with col3:
            st.empty()
    except Exception as e:
        st.error(f"Error rendering stock input: {str(e)}", icon="🚨")
        debug_session_state()

    if invalid_stocks:
        st.warning(f"Invalid stocks detected: {', '.join(invalid_stocks)}. Please enter valid stock symbols (e.g., TCS, INFY).", icon="⚠️")

# Define final_stock_list before any usage
all_stocks = list(set(valid_stocks + st.session_state.selected_stocks))
stock_mapping = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
mapped_stocks = [stock_mapping.get(stock, stock) for stock in all_stocks]
final_stock_list = [symbol + ".NS" if append_ns and symbol not in stock_mapping.values() else symbol for symbol in mapped_stocks]
logger.debug(f"Final stock list: {final_stock_list}")

with scanner_tab:
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #66FFCC;'>Analyze Stocks</h3>", unsafe_allow_html=True)
    
    timeframes_list = ["1d", "1wk", "1mo", "3mo"]
    periods_list = ["1mo", "6mo", "1y", "2y"]
    periods_intervals = {periods_list[i]: [timeframes_list[i]] for i in range(4)}
    tolerance = st.sidebar.slider("Price Tolerance for Nearby Zones (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

    # Make sure to initialize session state for dfs and zones_list if they don't exist
    if 'dfs' not in st.session_state:
        st.session_state.dfs = {}
    if 'zones_list' not in st.session_state:
        st.session_state.zones_list = {}

    # This is a simplified call, you might need to pass more arguments or handle state differently
    scanner.scanner_tab_ui(periods_intervals, timeframes_list, tolerance, zone_analysis_gv.generate_trade_recommendation, zone_analysis_gv.identify_zones, zone_analysis_gv.build_relationship_chart)


if final_stock_list:
    if not isinstance(st.session_state.analysis_results, dict):
        st.session_state.analysis_results = {}
    st.session_state.analysis_results = {k: v for k, v in st.session_state.analysis_results.items() if k in final_stock_list}

with backtesting_tab:
    st.header("Backtesting")
    timeframes_list_bt = ["1d", "1wk", "1mo", "3mo"]
    periods_list_bt = ["1mo", "6mo", "1y", "2y"]
    backtesting.backtesting_ui(final_stock_list, timeframes_list_bt, periods_list_bt, True, True, True, True, True, True)

with trade_charts_tab:
    st.header("Trade Charts")
    timeframes_list_tc = ["1d", "1wk", "1mo", "3mo"]
    periods_list_tc = ["1mo", "6mo", "1y", "2y"]
    backtesting.trade_charts_ui(final_stock_list, timeframes_list_tc, periods_list_tc, True, True, True, True, True, True)


with testing_tab:
    st.markdown("<h3 style='text-align: center; color: #66FFCC;'>Testing</h3>", unsafe_allow_html=True)
    if st.button("Download JSON Data"):
        if final_stock_list:
            try:
                json_data = generate_json_data(final_stock_list)
                json_str = json.dumps(json_data, default=str, indent=4)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="stock_data.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error generating JSON data: {str(e)}", icon="🚨")
        else:
            st.error("Please select or enter at least one valid stock before generating JSON.", icon="🚨")

def process_clean_charts():
    global chart_summaries
    chart_summaries = {}
    
    try:
        with summary_tab:
            if final_stock_list:
                if not isinstance(st.session_state.analysis_results, dict):
                    st.session_state.analysis_results = {}
                st.session_state.analysis_results = {k: v for k, v in st.session_state.analysis_results.items() if k in final_stock_list}
                
                all_data_intraday = fetch_and_process_data(final_stock_list, intraday_periods_intervals)
                all_data_short_term = fetch_and_process_data(final_stock_list, short_term_periods_intervals)
                all_data_long_term = fetch_and_process_data(final_stock_list, long_term_periods_intervals)

                clean_results = {}
                for stock in final_stock_list:
                    # Choose analysis method
                    if analysis_method == "Base/Following Candles":
                        identify_zones_func = identify_base_and_following_candles
                    else:
                        identify_zones_func = zone_analysis_gv.identify_zones
                    
                    for s, data_dict in all_data_intraday.items():
                        for (period, interval), data in data_dict.items():
                            key = f"{period}_{interval}"
                            if analysis_method == "Base/Following Candles":
                                base_rally_candles, base_drop_candles, is_clean = identify_zones_func(data, max_zones, max_overlap_pct, require_last_in_zone)
                                zones = base_rally_candles + base_drop_candles
                            else:
                                zones = identify_zones_func(data)
                                is_clean = True # For GV method, we don't have a 'clean' concept

                            st.session_state.analysis_results.setdefault(s, {})[key] = {
                                'data': data,
                                'zones': zones,
                                'is_clean': is_clean,
                                'category': 'Short-Term'
                            }
                            if is_clean and zones:
                                clean_results.setdefault(s, {'periods': [], 'intervals': []})
                                clean_results[s]['periods'].append(period)
                                clean_results[s]['intervals'].append(interval)
                    for s, data_dict in all_data_short_term.items():
                        for (period, interval), data in data_dict.items():
                            key = f"{period}_{interval}"
                            if analysis_method == "Base/Following Candles":
                                base_rally_candles, base_drop_candles, is_clean = identify_zones_func(data, max_zones, max_overlap_pct, require_last_in_zone)
                                zones = base_rally_candles + base_drop_candles
                            else:
                                zones = identify_zones_func(data)
                                is_clean = True 

                            st.session_state.analysis_results.setdefault(s, {})[key] = {
                                'data': data,
                                'zones': zones,
                                'is_clean': is_clean,
                                'category': 'Mid-Term'
                            }
                            if is_clean and zones:
                                clean_results.setdefault(s, {'periods': [], 'intervals': []})
                                clean_results[s]['periods'].append(period)
                                clean_results[s]['intervals'].append(interval)
                    for s, data_dict in all_data_long_term.items():
                        for (period, interval), data in data_dict.items():
                            key = f"{period}_{interval}"
                            if analysis_method == "Base/Following Candles":
                                base_rally_candles, base_drop_candles, is_clean = identify_zones_func(data, max_zones, max_overlap_pct, require_last_in_zone)
                                zones = base_rally_candles + base_drop_candles
                            else:
                                zones = identify_zones_func(data)
                                is_clean = True
                            st.session_state.analysis_results.setdefault(s, {})[key] = {
                                'data': data,
                                'zones': zones,
                                'is_clean': is_clean,
                                'category': 'Long-Term'
                            }
                            if is_clean and zones:
                                clean_results.setdefault(s, {'periods': [], 'intervals': []})
                                clean_results[s]['periods'].append(period)
                                clean_results[s]['intervals'].append(interval)
                
                # Store clean_results in session state for PDF generation
                st.session_state.clean_results = clean_results
                logger.debug(f"Charts results stored: {clean_results.keys()}")
                generate_summary_table(clean_results, st.session_state.filter_selected_intervals)

                with charts_tab:
                    displayed_stocks = False
                    for stock in final_stock_list:
                        display_stock = stock.replace(".NS", "")
                        clean_charts = {
                            k: v for k, v in st.session_state.analysis_results.get(stock, {}).items()
                            if v['is_clean'] and v['zones'] and k.split('_')[1] in st.session_state.filter_selected_intervals
                        }
                        stock_intervals = set(k.split('_')[1] for k in clean_charts.keys())
                        if set(st.session_state.filter_selected_intervals).issubset(stock_intervals):  # Strict AND condition
                            displayed_stocks = True
                            for key, details in clean_charts.items():
                                if '_' in key:
                                    period, interval = key.split('_')
                                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                                    st.markdown(f"<h4 style='font-size: 16px; color: #FAFAFA;'>{chart_labels.get(key, 'Chart')}</h4>", unsafe_allow_html=True)
                                    st.markdown(f"<h5 style='font-size: 14px; color: #D3D3D3;'>{display_stock} - {period} {interval}</h5>", unsafe_allow_html=True)
                                    plot_candlestick_chart(
                                        details['data'],
                                        [z for z in details['zones'] if z['type'] == 'demand'],
                                        [z for z in details['zones'] if z['type'] == 'supply'],
                                        f"{display_stock} - {period} {interval}",
                                        style='nightclouds'
                                    )
                    if not displayed_stocks:
                        st.markdown(f"<p style='font-size: 14px; color: #FF4D4D;'>No stocks found with all required intervals: {', '.join(st.session_state.filter_selected_intervals)}.</p>", unsafe_allow_html=True)
            else:
                with charts_tab:
                    st.error("Please select or enter at least one valid stock before generating charts.", icon="🚨")
    except Exception as e:
        st.error(f"Error processing charts: {str(e)}", icon="🚨")
        debug_session_state()

# --- (rest of the functions from app-full.py) --- 

if __name__ == "__main__":
    # The main execution block of the combined app
    # This will be very similar to the one in app-full.py, but with the new tabs
    process_clean_charts()
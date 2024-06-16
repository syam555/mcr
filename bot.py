import os
import requests
import time
import hashlib
import hmac
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = 'https://api.delta.exchange'

# Trading Parameters
RISK_TOLERANCE = 0.02  # 2% of capital
CAPITAL_ALLOCATION = 100  # Allocate $100 per trade
TARGET_PROFIT_PERCENTAGE = 0.5  # 50% target profit
MAXIMUM_LOSS_LIMIT = 0.25  # 25% maximum loss limit
TARGET_DELTA = 0  # Target delta for delta-neutral adjustments

def get_headers(payload):
    signature = hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    headers = {
        'api-key': API_KEY,
        'timestamp': str(int(time.time() * 1000)),
        'signature': signature,
        'Content-Type': 'application/json'
    }
    return headers

def fetch_ticker_symbols():
    url = f'{BASE_URL}/v2/tickers'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['result']
        tickers = [ticker['symbol'] for ticker in data if ticker['symbol'].endswith('USD') or ticker['symbol'].endswith('USDT')]
        logging.info(f"Fetched ticker symbols: {tickers}")
        return tickers
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding error: {json_err}")
        logging.error(f"Response content: {response.content}")
    return []

def fetch_price_data(symbol):
    url = f'{BASE_URL}/v2/tickers/{symbol}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Fetched price data for {symbol}: {json.dumps(data, indent=2)}")
        return data
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding error: {json_err}")
        logging.error(f"Response content: {response.content}")
    return None

def fetch_options_data():
    url = f'{BASE_URL}/v2/products'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        options_data = [product for product in data['result'] if 'BTC' in product['symbol'] or 'ETH' in product['symbol']]
        logging.info(f"Fetched options data: {json.dumps(options_data, indent=2)}")
        return options_data
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding error: {json_err}")
        logging.error(f"Response content: {response.content}")
    return None

def fetch_open_orders():
    url = f'{BASE_URL}/v2/orders/open'
    headers = get_headers("")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Fetched open orders: {json.dumps(data, indent=2)}")
        return data
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding error: {json_err}")
        logging.error(f"Response content: {response.content}")
    return None

def perform_technical_analysis(price_data):
    df = pd.DataFrame(price_data)
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['Bollinger_Upper'] = df['SMA20'] + 2 * df['close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA20'] - 2 * df['close'].rolling(window=20).std()
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])
    logging.info(f"Technical analysis results: {df.tail()}")
    return df

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_iv_percentile(options_data):
    if not options_data:
        logging.info("No options data available for IV calculation")
        return 0
    iv_values = [option['implied_volatility'] for option in options_data if 'implied_volatility' in option]
    if not iv_values:
        logging.info("No IV values found in options data")
        return 0
    iv_series = pd.Series(iv_values)
    current_iv = iv_series.iloc[-1]
    percentile = (iv_series < current_iv).sum() / len(iv_series)
    logging.info(f"Calculated IV percentile: {percentile * 100:.2f}%")
    return percentile

def select_strategy(iv_percentile):
    if iv_percentile < 0.2:
        return select_low_ivp_strategy()
    elif 0.2 <= iv_percentile <= 0.8:
        return select_medium_ivp_strategy()
    else:
        return select_high_ivp_strategy()

def select_low_ivp_strategy():
    strategies = [short_put_spread(), short_call_spread(), ratio_spread(), diagonal_spread()]
    selected_strategy = evaluate_strategies(strategies)
    logging.info(f"Selected low IVP strategy: {selected_strategy}")
    return selected_strategy

def select_medium_ivp_strategy():
    strategies = [iron_condor(), iron_butterfly()]
    selected_strategy = evaluate_strategies(strategies)
    logging.info(f"Selected medium IVP strategy: {selected_strategy}")
    return selected_strategy

def select_high_ivp_strategy():
    strategies = [strangle(), straddle()]
    selected_strategy = evaluate_strategies(strategies)
    logging.info(f"Selected high IVP strategy: {selected_strategy}")
    return selected_strategy

def evaluate_strategies(strategies):
    best_strategy = None
    best_score = float('-inf')
    
    for strategy in strategies:
        score = 0
        
        for option in strategy:
            if option['open_interest'] > 100 and option['volume'] > 50:
                score += 1
                
        if score > best_score:
            best_strategy = strategy
            best_score = score
            
    return best_strategy

def place_order(symbol, side, quantity, price, order_type='limit'):
    if not symbol or not side or quantity <= 0 or price <= 0:
        logging.error(f"Invalid order parameters: {locals()}")
        return None
    payload = json.dumps({
        'market': symbol,
        'side': side,
        'size': quantity,
        'price': price,
        'type': order_type
    })
    headers = get_headers(payload)
    try:
        response = requests.post(f'{BASE_URL}/v2/orders', headers=headers, data=payload)
        response.raise_for_status()
        logging.info(f"Placed order for {symbol}: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while placing order for {symbol}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred while placing order for {symbol}: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding error while placing order for {symbol}: {json_err}")
        logging.error(f"Response content: {response.content}")
    return None

def execute_with_retry(function, *args, retries=3, delay=5):
    for i in range(retries):
        try:
            result = function(*args)
            return result
        except Exception as e:
            logging.error(f"Error executing {function.__name__}: {e}")
            time.sleep(delay)
    return None

def short_put_spread():
    sell_strike = 'price_sell_put'
    buy_strike = 'price_buy_put'
    return [{'side': 'sell', 'strike': sell_strike, 'open_interest': 200, 'volume': 100, 'symbol': 'P-BTC-67400-170624'}, {'side': 'buy', 'strike': buy_strike, 'open_interest': 150, 'volume': 80, 'symbol': 'P-BTC-67000-170624'}]

def short_call_spread():
    sell_strike = 'price_sell_call'
    buy_strike = 'price_buy_call'
    return [{'side': 'sell', 'strike': sell_strike, 'open_interest': 180, 'volume': 90, 'symbol': 'C-BTC-67500-050724'}, {'side': 'buy', 'strike': buy_strike, 'open_interest': 160, 'volume': 70, 'symbol': 'C-BTC-68000-050724'}]

def ratio_spread():
    sell_strike = 'price_sell_ratio'
    buy_strike = 'price_buy_ratio'
    return [{'side': 'sell', 'strike': sell_strike, 'quantity': 2, 'open_interest': 250, 'volume': 120, 'symbol': 'P-BTC-67400-170624'}, {'side': 'buy', 'strike': buy_strike, 'quantity': 1, 'open_interest': 200, 'volume': 100, 'symbol': 'P-BTC-67000-170624'}]

def diagonal_spread():
    sell_strike = 'price_sell_diagonal'
    buy_strike = 'price_buy_diagonal'
    return [{'side': 'sell', 'strike': sell_strike, 'expiration': 'short_term', 'open_interest': 180, 'volume': 85, 'symbol': 'C-BTC-67500-050724'}, {'side': 'buy', 'strike': buy_strike, 'expiration': 'long_term', 'open_interest': 170, 'volume': 75, 'symbol': 'C-BTC-68000-050724'}]

def iron_condor():
    return [
        {'side': 'sell', 'strike': 'price_sell_put', 'open_interest': 220, 'volume': 110, 'symbol': 'P-BTC-67400-170624'},
        {'side': 'buy', 'strike': 'price_buy_put', 'open_interest': 200, 'volume': 100, 'symbol': 'P-BTC-67000-170624'},
        {'side': 'sell', 'strike': 'price_sell_call', 'open_interest': 210, 'volume': 105, 'symbol': 'C-BTC-67500-050724'},
        {'side': 'buy', 'strike': 'price_buy_call', 'open_interest': 190, 'volume': 95, 'symbol': 'C-BTC-68000-050724'}
    ]

def iron_butterfly():
    return [
        {'side': 'sell', 'strike': 'price_sell_atm_call', 'open_interest': 230, 'volume': 115, 'symbol': 'C-BTC-67500-050724'},
        {'side': 'sell', 'strike': 'price_sell_atm_put', 'open_interest': 230, 'volume': 115, 'symbol': 'P-BTC-67400-170624'},
        {'side': 'buy', 'strike': 'price_buy_otm_call', 'open_interest': 180, 'volume': 90, 'symbol': 'C-BTC-68000-050724'},
        {'side': 'buy', 'strike': 'price_buy_otm_put', 'open_interest': 180, 'volume': 90, 'symbol': 'P-BTC-67000-170624'}
    ]

def strangle():
    return [
        {'side': 'sell', 'strike': 'price_sell_otm_put', 'open_interest': 250, 'volume': 125, 'symbol': 'P-BTC-67400-170624'},
        {'side': 'sell', 'strike': 'price_sell_otm_call', 'open_interest': 250, 'volume': 125, 'symbol': 'C-BTC-67500-050724'}
    ]

def straddle():
    return [
        {'side': 'sell', 'strike': 'price_sell_atm_call', 'open_interest': 260, 'volume': 130, 'symbol': 'C-BTC-67500-050724'},
        {'side': 'sell', 'strike': 'price_sell_atm_put', 'open_interest': 260, 'volume': 130, 'symbol': 'P-BTC-67400-170624'}
    ]

def manage_trades(trades):
    open_orders = fetch_open_orders()
    if not open_orders:
        logging.info("No open orders found.")
        return
    
    for trade in trades:
        trade_exists = any(order['market'] == trade['symbol'] and order['side'] == trade['side'] for order in open_orders)
        
        if trade_exists:
            if should_roll(trade):
                roll_trade(trade)
            if should_hedge(trade):
                hedge_trade(trade)
            if should_adjust_delta(trade):
                adjust_delta(trade)
        else:
            placed_order = place_order(trade['symbol'], trade['side'], trade.get('quantity', 1), trade['strike'])
            if placed_order:
                logging.info(f"Successfully placed {trade['side']} order for {trade['symbol']} at {trade['strike']}")

def should_roll(trade):
    return trade.get('days_to_expiration', 0) < 7 and not trade.get('is_ITM', False)

def roll_trade(trade):
    logging.info(f"Rolling trade: {trade}")
    close_position(trade)
    new_trade = trade.copy()
    new_trade['expiration'] = trade.get('expiration', 0) + 30  # Example: roll to next month
    execute_with_retry(place_order, new_trade['symbol'], new_trade['side'], new_trade.get('quantity', 1), new_trade['strike'])

def should_hedge(trade):
    return trade.get('unrealized_loss', 0) > (CAPITAL_ALLOCATION * MAXIMUM_LOSS_LIMIT)

def hedge_trade(trade):
    logging.info(f"Hedging trade: {trade}")
    hedge_position = trade.copy()
    hedge_position['side'] = 'buy' if trade['side'] == 'sell' else 'sell'
    execute_with_retry(place_order, hedge_position['symbol'], hedge_position['side'], hedge_position.get('quantity', 1), hedge_position['strike'])

def should_adjust_delta(trade):
    return abs(trade.get('delta', 0)) > TARGET_DELTA

def adjust_delta(trade):
    logging.info(f"Adjusting delta for trade: {trade}")
    if trade.get('delta', 0) > 0:
        execute_with_retry(place_order, trade['symbol'], 'sell', trade.get('quantity', 1), trade['strike'])
    else:
        execute_with_retry(place_order, trade['symbol'], 'buy', trade.get('quantity', 1), trade['strike'])

def close_position(trade):
    payload = json.dumps({
        'market': trade['symbol'],
        'side': 'close',
        'size': trade.get('quantity', 1),
        'price': trade['strike'],
        'type': 'limit'
    })
    headers = get_headers(payload)
    try:
        response = requests.post(f'{BASE_URL}/v2/orders', headers=headers, data=payload)
        response.raise_for_status()
        logging.info(f"Closed position for {trade['symbol']}: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred while closing position for {trade['symbol']}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred while closing position for {trade['symbol']}: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding error while closing position for {trade['symbol']}: {json_err}")
        logging.error(f"Response content: {response.content}")
    return None

def advanced_technical_analysis(ticker):
    price_data = fetch_price_data(ticker)
    if not price_data or 'result' not in price_data:
        logging.info(f"No valid price data for {ticker}")
        return None
    
    result = price_data['result']
    
    # Flatten the nested dictionary
    flattened_data = {k: [v] for k, v in result.items() if not isinstance(v, dict)}
    
    try:
        df = pd.DataFrame(flattened_data)
    except ValueError as e:
        logging.error(f"Error converting price data to DataFrame for {ticker}: {e}")
        return None

    # Implement technical indicators and analysis
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['SMA200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])

    logging.info(f"Technical analysis results for {ticker}: {df.tail()}")

    # Example leverage strategy based on technical indicators
    if df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1] and df['RSI'].iloc[-1] < 70 and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        # Buy signal
        placed_order = place_order(ticker, 'buy', 1, df['close'].iloc[-1], 'market')
        if placed_order:
            logging.info(f"Buy signal for {ticker} executed successfully.")
    elif df['SMA50'].iloc[-1] < df['SMA200'].iloc[-1] and df['RSI'].iloc[-1] > 30 and df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
        # Sell signal
        placed_order = place_order(ticker, 'sell', 1, df['close'].iloc[-1], 'market')
        if placed_order:
            logging.info(f"Sell signal for {ticker} executed successfully.")

def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def trade_options():
    options_data = fetch_options_data()
    if not options_data:
        logging.info("No options data available.")
        return

    for option in options_data:
        symbol = option['symbol']
        if 'BTC' in symbol or 'ETH' in symbol:
            if symbol.startswith('P-'):
                trade_put_option(symbol)
            elif symbol.startswith('C-'):
                trade_call_option(symbol)

def trade_put_option(symbol):
    # Logic to trade put options
    logging.info(f"Trading PUT option: {symbol}")
    # Example logic for trading PUT options
    placed_order = place_order(symbol, 'sell', 1, get_market_price(symbol))
    if placed_order:
        logging.info(f"Successfully placed PUT option order for {symbol}")

def trade_call_option(symbol):
    # Logic to trade call options
    logging.info(f"Trading CALL option: {symbol}")
    # Example logic for trading CALL options
    placed_order = place_order(symbol, 'sell', 1, get_market_price(symbol))
    if placed_order:
        logging.info(f"Successfully placed CALL option order for {symbol}")

def get_market_price(symbol):
    # Placeholder function to get the current market price for the symbol
    price_data = fetch_price_data(symbol)
    if price_data:
        return price_data['result']['close']
    return None

def monitor_market():
    logging.info("Started monitoring market")
    tickers = fetch_ticker_symbols()
    while True:
        for ticker in tickers:
            logging.info(f"Fetching price data for {ticker}")
            price_data = execute_with_retry(fetch_price_data, ticker)
            
            if not price_data:
                continue

            # Only fetch options data for BTC and ETH
            if ticker in ['BTCUSD', 'ETHUSD']:
                logging.info(f"Fetching options data for {ticker}")
                options_data = fetch_options_data()
                iv_percentile = calculate_iv_percentile(options_data)
                strategy = select_strategy(iv_percentile)
                manage_trades(strategy)
                logging.info(f"{ticker} Strategy: {strategy}")
                continue

            # Implement trading logic based on fetched price data
            advanced_technical_analysis(ticker)

        time.sleep(300)  # Check every 5 minutes

def main():
    logging.info("Bot started successfully")
    monitor_market()

if __name__ == "__main__":
    main()

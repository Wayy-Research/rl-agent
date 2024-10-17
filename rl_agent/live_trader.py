import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_processing import add_technical_indicators
import logging
from dotenv import load_dotenv
import os
import time
from stable_baselines3 import PPO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

import signal
from collections import deque

class AlpacaLiveTrader:
    def __init__(self, model_path):
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        self.model = self.load_model(model_path)
        self.symbol = 'BTCUSD'  # Default to Bitcoin, can be changed
        self.running = False
        self.trade_history = deque(maxlen=100)  # Store last 100 trades
        signal.signal(signal.SIGINT, self.stop_trading)

    def load_model(self, model_path):
        return PPO.load(model_path)

    def get_account(self):
        return self.api.get_account()

    def get_position(self):
        try:
            return self.api.get_position(self.symbol)
        except:
            return None

    def get_bars(self, timeframe='1Min', limit=100):
        bars = self.api.get_crypto_bars(self.symbol, timeframe, limit=limit).df
        bars = bars[bars.exchange == 'CBSE']  # Filter for Coinbase exchange
        bars = bars.reset_index().rename(columns={"timestamp": "timestamp", "symbol": "tic"})
        return bars

    def preprocess_data(self, bars):
        preprocessed_bars = add_technical_indicators(bars)
        preprocessed_bars.dropna(inplace=True)
        return preprocessed_bars

    def get_state(self, bars):
        # Convert the last row of bars into the state vector expected by the model
        last_row = bars.iloc[-1]
        state = [
            last_row['open'],
            last_row['high'],
            last_row['low'],
            last_row['close'],
            last_row['volume'],
            # Add other technical indicators here
        ]
        return np.array(state)

    def execute_trade(self, action):
        current_position = self.get_position()
        account = self.get_account()
        buying_power = float(account.buying_power)
        
        try:
            if action > 0:  # Buy
                if current_position is None or float(current_position.qty) == 0:
                    buy_amount = min(abs(action) * buying_power, buying_power * 0.1)  # Use at most 10% of buying power
                    qty = buy_amount / float(self.get_current_price())
                    order = self.api.submit_order(
                        symbol=self.symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    logging.info(f'Buy order submitted: {order}')
                    self.trade_history.append(('buy', qty, float(self.get_current_price())))
            elif action < 0:  # Sell
                if current_position is not None and float(current_position.qty) > 0:
                    sell_qty = min(abs(action) * float(current_position.qty), float(current_position.qty))
                    order = self.api.submit_order(
                        symbol=self.symbol,
                        qty=sell_qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    logging.info(f'Sell order submitted: {order}')
                    self.trade_history.append(('sell', sell_qty, float(self.get_current_price())))
        except Exception as e:
            logging.error(f'Error executing trade: {e}')

    def get_current_price(self):
        bars = self.api.get_crypto_bars(self.symbol, '1Min', limit=1).df
        return bars['close'].iloc[-1]

    def calculate_performance(self):
        if len(self.trade_history) < 2:
            return None
        
        initial_price = self.trade_history[0][2]
        final_price = self.trade_history[-1][2]
        returns = (final_price - initial_price) / initial_price
        
        return {
            'returns': returns,
            'num_trades': len(self.trade_history)
        }

    def stop_trading(self, signum, frame):
        logging.info('Stopping trading...')
        self.running = False

    def run(self):
        self.running = True
        backoff = 1
        while self.running:
            try:
                bars = self.get_bars()
                preprocessed_bars = self.preprocess_data(bars)
                state = self.get_state(preprocessed_bars)
                action, _ = self.model.predict(state)
                self.execute_trade(action)
                
                account = self.get_account()
                logging.info(f'Account status - Cash: ${account.cash}, Portfolio Value: ${account.portfolio_value}')
                
                performance = self.calculate_performance()
                if performance:
                    logging.info(f'Performance - Returns: {performance["returns"]:.2%}, Number of trades: {performance["num_trades"]}')
                
                backoff = 1
                time.sleep(60)
            except Exception as e:
                logging.error(f'Error in trading loop: {e}')
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)

if __name__ == "__main__":
    model_path = 'path/to/your/trained/model.zip'
    trader = AlpacaLiveTrader(model_path)
    trader.run()
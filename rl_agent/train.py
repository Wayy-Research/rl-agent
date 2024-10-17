import argparse
import numpy as np
import pandas as pd
from trading_env import CryptoTradingEnv
from data_processing import download_data, add_technical_indicators, data_split
from stable_baselines3 import PPO

class CryptoTrader:
    def __init__(self, env, model_name="ppo_crypto_trader", mode="backtest"):
        self.env = env
        self.model_name = model_name
        self.mode = mode
        self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_name)

    def backtest(self, test_data):
        obs = self.env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            # TODO: Log results or store for later analysis

    def live_trade(self):
        # TODO: Implement live trading logic using Alpaca API
        pass

def main(args):
    # Load and preprocess data
    data = download_data(args.symbols, args.start_date, args.end_date)
    preprocessed_data = add_technical_indicators(data)
    preprocessed_data.dropna(inplace=True)

    # Create environment
    env = CryptoTradingEnv(
        df=preprocessed_data,
        crypto_dim=args.crypto_dim,
        hmax=args.hmax,
        initial_amount=args.initial_amount,
        buy_cost_pct=args.buy_cost_pct,
        sell_cost_pct=args.sell_cost_pct,
        reward_scaling=args.reward_scaling,
        state_space=args.state_space,
        action_space=args.action_space,
        tech_indicator_list=args.tech_indicators,
        volatility_threshold=args.volatility_threshold
    )

    # Create and train the agent
    trader = CryptoTrader(env, model_name=args.model_name, mode=args.mode)
    
    if args.mode == "train":
        trader.train(total_timesteps=args.total_timesteps)
    elif args.mode == "backtest":
        trader.backtest(test_data=preprocessed_data)  # Assume we use the same data for simplicity
    elif args.mode == "live":
        trader.live_trade()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "backtest", "live"], default="train")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for data download")
    parser.add_argument("--end_date", type=str, required=True, help="End date for data download")
    parser.add_argument("--symbols", nargs="+", default=None, help="List of crypto symbols to download")
    parser.add_argument("--model_name", type=str, default="ppo_crypto_trader")
    parser.add_argument("--crypto_dim", type=int, default=1)
    parser.add_argument("--hmax", type=float, default=100)
    parser.add_argument("--initial_amount", type=float, default=1000)
    parser.add_argument("--buy_cost_pct", type=float, default=0.001)
    parser.add_argument("--sell_cost_pct", type=float, default=0.001)
    parser.add_argument("--reward_scaling", type=float, default=1e-4)
    parser.add_argument("--state_space", type=int, default=10)
    parser.add_argument("--action_space", type=int, default=3)
    parser.add_argument("--tech_indicators", nargs="+", default=["macd", "rsi", "cci", "adx"])
    parser.add_argument("--volatility_threshold", type=float, default=None)
    parser.add_argument("--total_timesteps", type=int, default=10000)

    args = parser.parse_args()
    main(args)

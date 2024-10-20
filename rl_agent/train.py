import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import CryptoTradingEnv
from data_processing import download_data, add_technical_indicators, data_split
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class PerformanceLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PerformanceLoggingCallback, self).__init__(verbose)
        self.performance_log = []
        self.episode_count = 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.episode_count += 1
        performance_metrics = self.training_env.envs[0].calculate_backtest_metrics()
        self.performance_log.append(performance_metrics)
        
        print(f"Episode {self.episode_count} Results:")
        print(f"Total Return: {performance_metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {performance_metrics['total_trades']}")
        print(f"Win Rate: {performance_metrics['win_rate']:.2%}")
        
        
        return True

class CryptoTrader:
    def __init__(self, env, model_name="ppo_crypto_trader", mode="train"):
        self.env = env
        self.model_name = model_name
        self.mode = mode
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.performance_callback = PerformanceLoggingCallback()

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps, callback=self.performance_callback)
        self.model.save(self.model_name)
        self.plot_performance()
        print(f"Training completed. Model saved as {self.model_name}")

    def plot_performance(self):
        performance_log = self.performance_callback.performance_log
        episodes = range(len(performance_log))

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [log['total_return'] for log in performance_log], label='Total Return')
        plt.plot(episodes, [log['sharpe_ratio'] for log in performance_log], label='Sharpe Ratio')
        plt.plot(episodes, [log['max_drawdown'] for log in performance_log], label='Max Drawdown')
        plt.xlabel('Episode')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics During Training')
        plt.legend()
        plt.savefig(f"{self.model_name}_performance.png")
        plt.close()

    def backtest(self, test_data):
        obs = self.env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        
        metrics = self.env.calculate_backtest_metrics()
        print("Backtest Results:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")

    def evaluate_on_new_data(self, new_data):
        metrics = self.env.evaluate_on_new_data(new_data)
        print("Evaluation Results on New Data:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")

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
        volatility_threshold=args.volatility_threshold,
        mode=args.mode
    )

    # Create and train the agent
    trader = CryptoTrader(env, model_name=args.model_name, mode=args.mode)
    
    if args.mode == "train":
        trader.train(total_timesteps=args.total_timesteps)
    elif args.mode == "backtest":
        trader.backtest(test_data=preprocessed_data)  # Assume we use the same data for simplicity
    elif args.mode == "evaluate":
        # Load new data for evaluation
        new_data = download_data(args.symbols, args.eval_start_date, args.eval_end_date)
        new_preprocessed_data = add_technical_indicators(new_data)
        new_preprocessed_data.dropna(inplace=True)
        trader.evaluate_on_new_data(new_preprocessed_data)
    elif args.mode == "live":
        trader.live_trade()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "backtest", "evaluate", "live"], default="train")
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
    parser.add_argument("--tech_indicators", nargs="+", default=["macd", "rsi_30", "cci_30", "adx"])
    parser.add_argument("--volatility_threshold", type=float, default=None)
    parser.add_argument("--total_timesteps", type=int, default=10000)
    parser.add_argument("--eval_start_date", type=str, help="Start date for evaluation data")
    parser.add_argument("--eval_end_date", type=str, help="End date for evaluation data")
    
    args = parser.parse_args()
    main(args)

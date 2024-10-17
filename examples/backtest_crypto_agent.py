import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from rl_agent.trading_env import CryptoTradingEnv
from stable_baselines3 import PPO
from rl_agent.data_processing import download_and_process_data

def train_agent(env, total_timesteps=10000):
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

def backtest_agent(env, model):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

    # Calculate and print backtest metrics
    metrics = env.calculate_backtest_metrics()
    print('Backtest Results:')
    print(f'Total Return: {metrics["total_return"]:.2%}')
    print(f'Sharpe Ratio: {metrics["sharpe_ratio"]:.2f}')
    print(f'Max Drawdown: {metrics["max_drawdown"]:.2%}')

def main():
    # Download and process data
    df = download_and_process_data(start_date='2022-01-01', end_date='2023-01-01', symbols=['BTC/USD'])

    # Create training environment
    train_env = CryptoTradingEnv(
        df=df,
        crypto_dim=1,
        hmax=100,
        initial_amount=10000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=7,  # Adjust based on your actual state space
        action_space=1,
        tech_indicator_list=['rsi', 'macd', 'cci', 'adx'],
        mode='train'
    )

    # Train the agent
    trained_model = train_agent(train_env)

    # Create backtesting environment
    backtest_env = CryptoTradingEnv(
        df=df,
        crypto_dim=1,
        hmax=100,
        initial_amount=10000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=7,  # Adjust based on your actual state space
        action_space=1,
        tech_indicator_list=['rsi', 'macd', 'cci', 'adx'],
        mode='backtest'
    )

    # Run backtest
    backtest_agent(backtest_env, trained_model)

if __name__ == '__main__':
    main()

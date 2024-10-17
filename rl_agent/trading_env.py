import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common import logger


class CryptoTradingEnv(gym.Env):
    """A cryptocurrency trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        crypto_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        volatility_threshold=None,
        risk_indicator_col="volatility",
        make_plots=False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode='train',
        iteration="",
    ):
        self.day = day
        self.df = df.reset_index(drop=True)  # Reset index to ensure it starts from 0
        self.crypto_dim = crypto_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.iloc[self.day]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.volatility_threshold = volatility_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Columns: {self.df.columns}")
        print(f"State space: {self.state_space}")
        print(f"Crypto dim: {self.crypto_dim}")
        print(f"Tech indicators: {self.tech_indicator_list}")
        
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.volatility = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.timestamp_memory = [self._get_timestamp()]
        # self.reset()
        self._seed()

        if self.mode == 'backtest':
            self.backtest_returns = []
            self.backtest_portfolio_values = []

    def _sell_crypto(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                if self.state[index + self.crypto_dim + 1] > 0:
                    # Sell only if current crypto amount is > 0
                    sell_amount = min(abs(action), self.state[index + self.crypto_dim + 1])
                    sell_value = self.state[index + 1] * sell_amount * (1 - self.sell_cost_pct)
                    # update balance
                    self.state[0] += sell_value
                    self.state[index + self.crypto_dim + 1] -= sell_amount
                    self.cost += self.state[index + 1] * sell_amount * self.sell_cost_pct
                    self.trades += 1
                else:
                    sell_amount = 0
            else:
                sell_amount = 0

            return sell_amount

        # perform sell action based on the sign of the action
        if self.volatility_threshold is not None:
            if self.volatility >= self.volatility_threshold:
                if self.state[index + 1] > 0:
                    # if volatility goes over threshold, just clear out all positions
                    if self.state[index + self.crypto_dim + 1] > 0:
                        sell_amount = self.state[index + self.crypto_dim + 1]
                        sell_value = self.state[index + 1] * sell_amount * (1 - self.sell_cost_pct)
                        # update balance
                        self.state[0] += sell_value
                        self.state[index + self.crypto_dim + 1] = 0
                        self.cost += self.state[index + 1] * sell_amount * self.sell_cost_pct
                        self.trades += 1
                    else:
                        sell_amount = 0
                else:
                    sell_amount = 0
            else:
                sell_amount = _do_sell_normal()
        else:
            sell_amount = _do_sell_normal()

        return sell_amount

    def _buy_crypto(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] / self.state[index + 1]
                # update balance
                buy_amount = min(available_amount, action)
                buy_cost = self.state[index + 1] * buy_amount * (1 + self.buy_cost_pct)
                self.state[0] -= buy_cost

                self.state[index + self.crypto_dim + 1] += buy_amount

                self.cost += self.state[index + 1] * buy_amount * self.buy_cost_pct
                self.trades += 1
            else:
                buy_amount = 0

            return buy_amount

        # perform buy action based on the sign of the action
        if self.volatility_threshold is None:
            buy_amount = _do_buy()
        else:
            if self.volatility < self.volatility_threshold:
                buy_amount = _do_buy()
            else:
                buy_amount = 0

        return buy_amount

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index) - 1
        
        # Calculate begin_total_asset and end_total_asset at the start of the method
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.crypto_dim + 1)])
            * np.array(self.state[(self.crypto_dim + 1) : (self.crypto_dim * 2 + 1)])
        )
        
        if not self.terminal:
            # ... (existing code for updating state and performing actions)
            
            # Update end_total_asset after state update
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.crypto_dim + 1)])
                * np.array(self.state[(self.crypto_dim + 1) : (self.crypto_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.timestamp_memory.append(self._get_timestamp())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            
            if self.mode == 'backtest':
                self.backtest_returns.append(self.reward)
                self.backtest_portfolio_values.append(end_total_asset)
        
        if self.terminal:
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.crypto_dim + 1)])
                * np.array(self.state[(self.crypto_dim + 1) : (self.crypto_dim * 2 + 1)])
            )
            #

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.crypto_dim + 1)])
                * np.array(
                    self.previous_state[(self.crypto_dim + 1) : (self.crypto_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.iloc[self.day]
        self.volatility = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.timestamp_memory = [self._get_timestamp()]

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        state = [
            self.initial_amount,  # Account balance
            self.data['close'],   # Close price
        ]
        state.extend([0] * self.crypto_dim)  # Owned crypto
        state.extend([self.data[tech] for tech in self.tech_indicator_list])  # Technical indicators
        
        print(f"Initiated state: {state}")
        print(f"Initiated state length: {len(state)}")
        
        # Pad or truncate the state to match state_space
        if len(state) < self.state_space:
            state.extend([0] * (self.state_space - len(state)))
        elif len(state) > self.state_space:
            state = state[:self.state_space]
        
        print(f"Final initiated state: {state}")
        print(f"Final initiated state length: {len(state)}")
        
        return np.array(state)

    def _update_state(self):
        state = [
            self.state[0],  # Account balance
            self.data['close'],  # Close price
        ]
        state.extend(self.state[2:2+self.crypto_dim])  # Owned crypto
        state.extend([self.data[tech] for tech in self.tech_indicator_list])  # Technical indicators
        
        print(f"Updated state: {state}")
        print(f"Updated state length: {len(state)}")
        
        # Pad or truncate the state to match state_space
        if len(state) < self.state_space:
            state.extend([0] * (self.state_space - len(state)))
        elif len(state) > self.state_space:
            state = state[:self.state_space]
        
        print(f"Final updated state: {state}")
        print(f"Final updated state length: {len(state)}")
        
        return np.array(state)

    def _get_timestamp(self):
        if isinstance(self.data, pd.DataFrame):
            if len(self.df.tic.unique()) > 1:
                timestamp = self.data['timestamp'].iloc[0]
            else:
                timestamp = self.data['timestamp'].iloc[0]
        else:  # If self.data is a Series
            timestamp = self.data.name if self.data.name else pd.Timestamp(self.day)
        return timestamp

    def save_asset_memory(self):
        timestamp_list = self.timestamp_memory
        asset_list = self.asset_memory
        # print(len(timestamp_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"timestamp": timestamp_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # timestamp and close price length must match actions length
            timestamp_list = self.timestamp_memory[:-1]
            df_timestamp = pd.DataFrame(timestamp_list)
            df_timestamp.columns = ["timestamp"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_timestamp.timestamp
            # df_actions = pd.DataFrame({'timestamp':timestamp_list,'actions':action_list})
        else:
            timestamp_list = self.timestamp_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"timestamp": timestamp_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def calculate_backtest_metrics(self):
        if self.mode != 'backtest':
            raise ValueError('Metrics calculation is only available in backtest mode')
        
        returns = np.array(self.backtest_returns)
        portfolio_values = np.array(self.backtest_portfolio_values)
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

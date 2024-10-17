# Cryptocurrency Trading Agent with Reinforcement Learning

## Description

This project implements a reinforcement learning-based trading agent for cryptocurrencies. It uses the Alpaca API for data retrieval and trading, and implements a custom gym environment for training the agent. The project includes modules for creating a trading environment, processing data, training models, and performing live trading.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, follow these steps:

```bash
git clone https://github.com/wayy-research/rl-agent.git
cd rl-agent
pip install -r requirements.txt
pip install .
```

Make sure to set up your Alpaca API credentials in a `.env` file:

```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Usage

To train the agent:

```python
python rl_agent/train.py --mode train --start_date 2020-01-01 --end_date 2023-01-01 --symbols BTC/USD ETH/USD
```

To backtest the trained agent:

```python
python rl_agent/train.py --mode backtest --start_date 2023-01-01 --end_date 2023-12-31 --symbols BTC/USD ETH/USD
```

To run live trading:

```python
python rl_agent/live_trader.py
```

## Features

- Custom OpenAI Gym environment for cryptocurrency trading
- Data processing and technical indicator calculation
- Integration with Alpaca API for real-time data and trading
- Reinforcement learning model training using Stable Baselines3
- Backtesting capabilities
- Live trading functionality

## Project Structure

- `rl_agent/`
  - `trading_env.py`: Custom OpenAI Gym environment for cryptocurrency trading
  - `data_processing.py`: Functions for data download and preprocessing
  - `train.py`: Script for training and backtesting the RL agent
  - `live_trader.py`: Implementation of live trading using the trained agent
- `requirements.txt`: List of project dependencies
- `setup.py`: Setup script for the project

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes and commit them (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
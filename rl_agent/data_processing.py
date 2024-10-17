import os
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
from stockstats import StockDataFrame as sdf
from finrl.meta.data_processor import DataProcessor
# alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.data.timeframe import TimeFrame
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest


load_dotenv()
key =os.environ.get('ALPACA_KEY')
secret = os.environ.get('ALPACA_SECRET')

def get_crypto_symbols() -> List[str]:
    """
    Retrieve a list of available cryptocurrency symbols from Alpaca.

    Returns:
        List[str]: A list of cryptocurrency symbols.
    """
    trading_client = TradingClient(key, secret, paper=True)
    # search for crypto assets
    search_params = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
    assets = trading_client.get_all_assets(search_params)
    return [asset.symbol for asset in assets]

symbols = get_crypto_symbols()


TRAIN_START_DATE = '2020-05-01' 
TRAIN_END_DATE= '2023-07-01' 
TRADE_START_DATE = '2023-07-02' 
TRADE_END_DATE = '2024-09-01'
TIME_INTERVAL = '1d'

def download_data(symbols: List[str], start_date: str = TRAIN_START_DATE, end_date: str = TRADE_END_DATE) -> pd.DataFrame:
    """
    Download cryptocurrency data for the specified symbols and date range.

    Args:
        symbols (List[str]): List of cryptocurrency symbols to download data for.
        start_date (str, optional): Start date for the data range. Defaults to TRAIN_START_DATE.
        end_date (str, optional): End date for the data range. Defaults to TRADE_END_DATE.

    Returns:
        pd.DataFrame: A DataFrame containing the downloaded cryptocurrency data.
    """
    crypto_client = CryptoHistoricalDataClient(key, secret)
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols[:10],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )

    # Fetch the data
    df = crypto_client.get_crypto_bars(request_params).df

    # convert timezone to NY
    NY = "America/New_York"
    df.index = df.index.set_levels(df.index.levels[1].tz_convert(NY), level=1)

    df = df.reset_index().rename(columns={"index": "timestamp", "symbol": "tic"})

    df = df.sort_values(by=["tic", "timestamp"], ascending=True)

    return df



def add_technical_indicators(
    df: pd.DataFrame,
    INDICATORS: List[str] = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "adx",
        "close_30_sma",
        "close_60_sma",
    ],
) -> pd.DataFrame:
    """
    Add technical indicators to the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing cryptocurrency data.
        INDICATORS (List[str], optional): List of technical indicators to add. Defaults to a predefined list.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    for tic in df['tic'].unique():
        stock = sdf.retype(df[df['tic'] == tic].copy())
        for indicator in INDICATORS:
            df.loc[df['tic'] == tic, indicator] = stock[indicator]
    
    df.dropna(inplace=True)
    return df

def data_split(df: pd.DataFrame, start: str, end: str, target_date_col: str = "timestamp") -> pd.DataFrame:
    """
    Split the input DataFrame based on the specified date range.

    Args:
        df (pd.DataFrame): Input DataFrame containing cryptocurrency data.
        start (str): Start date for the split.
        end (str): End date for the split.
        target_date_col (str, optional): Name of the column containing dates. Defaults to "timestamp".

    Returns:
        pd.DataFrame: DataFrame containing data within the specified date range.
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data
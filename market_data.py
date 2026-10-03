import pandas as pd
import yfinance as yf

from strategy_core import INTERVAL


class YahooMarketData:
    """Current development data provider. Replaceable later without strategy changes."""

    name = "yfinance"

    def history(self, symbol, period="60d", interval=INTERVAL):
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data.dropna().copy()

    def day(self, symbol, date_text, interval=INTERVAL):
        start = pd.Timestamp(date_text)
        end = start + pd.Timedelta(days=1)

        data = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data.dropna().copy()


DEFAULT_PROVIDER = YahooMarketData()

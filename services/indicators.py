import pandas as pd
import pandas_ta as ta

from services.schema import IntradayIndicator, HTFIndicator


# ---------- HELPER ----------
def candles_to_df(candles):
    df = pd.DataFrame([c.dict() for c in candles])

    # DO NOT REMOVE THIS
    df = df.sort_values("timestamp")

    df.set_index("timestamp", inplace=True)

    return df


# ---------- INTRADAY ----------
def compute_intraday_features(candles) -> list[IntradayIndicator]:
    df = candles_to_df(candles)

    df["ema_20"] = ta.ema(df["close"], length=20)

    df["rsi_7"] = ta.rsi(df["close"], length=7)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    df = df.tail(50)

    return [
        IntradayIndicator(
            timestamp=idx.to_pydatetime(),
            close=row["close"],
            ema_20=row.get("ema_20"),
            rsi_7=row.get("rsi_7"),
            rsi_14=row.get("rsi_14"),
            macd=row.get("macd"),
            macd_signal=row.get("macd_signal"),
        )
        for idx, row in df.iterrows()
    ]


# ---------- HTF ----------
def compute_htf_features(candles) -> list[HTFIndicator]:
    df = candles_to_df(candles)

    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_cross"] = df["ema_20"] > df["ema_50"]

    df["rsi"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    df = df.tail(50)

    return [
        HTFIndicator(
            timestamp=idx.to_pydatetime(),
            close=row["close"],
            ema_20=row.get("ema_20"),
            ema_50=row.get("ema_50"),
            ema_cross=row.get("ema_cross"),
            rsi=row.get("rsi"),
            macd=row.get("macd"),
            macd_signal=row.get("macd_signal"),
            atr=row.get("atr"),
        )
        for idx, row in df.iterrows()
    ]


# ---------- WRAPPER ----------
def build_multi_tf_features(intraday_candles, htf_candles):
    return {
        "intraday": compute_intraday_features(intraday_candles),
        "higher_tf": compute_htf_features(htf_candles),
    }
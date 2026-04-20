from hl_market import MarketService
from indicators import build_multi_tf_features
import time

def main():
    symbol = "BTC"
    market = MarketService(testnet=True)

    # ---------- FETCH ----------
    t0 = time.time()

    intraday_candles = market.get_candles(symbol, interval="5m", limit=100)
    htf_candles = market.get_candles(symbol, interval="4h", limit=100)

    t1 = time.time()

    print(f"\nFetch time: {t1 - t0:.4f} sec")

    # ---------- COMPUTE ----------
    t2 = time.time()

    features = build_multi_tf_features(intraday_candles, htf_candles)

    t3 = time.time()

    print(f"Compute time: {t3 - t2:.4f} sec")

    # ---------- TOTAL ----------
    print(f"Total time: {t3 - t0:.4f} sec")

    # ---------- OUTPUT ----------
    print("\n===== INTRADAY (LAST 1) =====")
    print(features["intraday"][-1])

    print("\n===== HTF (LAST 1) =====")
    print(features["higher_tf"][-1])

if __name__ == "__main__":
    main()
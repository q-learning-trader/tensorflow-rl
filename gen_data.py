import sys

import numpy as np
import pandas as pd
import ta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def gen_data(file_path="gbpjpy15.csv"):
    try:
        print("load file")
        df = pd.read_csv(file_path)
    except:
        print("Use 'python gen_data.py 'file_path''")
        return

    df["Close1"] = df["Close"] * 100

    # returns = np.array(np.log(df["Close"] / df["Close"].shift(1))).reshape((-1,1)) * 1000
    # ma = np.array(np.log(ta.trend.ema(df["Close1"], 14) / ta.trend.ema(df["Close1"], 7))).reshape((-1,1)) * 1000
    # macd = np.array(ta.trend.macd_diff(df["Close1"])).reshape((-1,1))
    rsi = np.array(ta.momentum.rsi(df["Close"]) - ta.momentum.rsi(df["Close"], 7)).reshape((-1,1))
    stoch = np.array(ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"]) - ta.momentum.stoch(df["High"], df["Low"], df["Close"])).reshape((-1,1))

    x = np.concatenate([rsi, stoch], -1)
    # x = returns

    y = np.array(df[["Open"]])
    atr = np.array(ta.volatility.average_true_range(df["High"], df["Low"], df["Close"]))
    high = np.array(df[["High"]])
    low = np.array(df[["Low"]])

    print("gen time series data")
    gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x, y, 10)
    x = []
    y = []
    for i in gen:
        x.extend(i[0].tolist())
        y.extend(i[1].tolist())
    x = np.asanyarray(x)[100:]
    y = np.asanyarray(y)[100:]
    atr = atr[-len(y):].reshape((-1, 1))
    scale_atr = atr
    high = high[-len(y):].reshape((-1, 1))
    low = low[-len(y):].reshape((-1, 1))

    np.save("x", x)
    np.save("target", np.array([y, atr, scale_atr, high, low]))

    print("done\n")

if __name__ == "__main__":
    argv = sys.argv
    print(argv[1])
    gen_data(argv[1])

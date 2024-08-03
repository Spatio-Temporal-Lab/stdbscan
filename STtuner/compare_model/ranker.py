import datetime
import time

import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from compare_model.xgb_rank_model import XGBClassifierModel


class Ranker:
    def __init__(self, learning_rate=0.1, colsample_bytree=0.9, eta=0.05, max_depth=6, n_estimators=110,
                 subsample=0.75):
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.eta = eta
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.rank_model = None
        self.pred_model = None

    def prepare_data(self, file):
        df = pd.read_csv(file)
        return self.format_data(df)

    def format_data(self, df):
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df = df.dropna(subset=['result'])
        df.loc[:, 'result'] = df['result'].astype(int)
        df.reset_index(drop=True, inplace=True)

        def convert_unit(value):
            if isinstance(value, str):
                if value.endswith('g'):
                    return float(value[:-1])
                elif value.endswith('m'):
                    return float(value[:-1])
                elif value.endswith('k'):
                    return float(value[:-1])
            return float(value)

        columns_unit = [
            'spark.executor.memory', 'spark.memory.offHeap.size', 'spark.shuffle.file.buffer',
            'spark.reducer.maxSizeInFlight', 'spark.speculation.interval', 'spark.broadcast.blockSize',
            'spark.io.compression.lz4.blockSize', 'spark.io.compression.snappy.blockSize',
            'spark.kryoserializer.buffer.max'
        ]
        for col in columns_unit:
            df[col] = df[col].apply(convert_unit)

        columns_bool = [
            'spark.memory.offHeap.enabled', 'spark.speculation', 'spark.kryo.referenceTracking',
            'spark.shuffle.compress', 'spark.shuffle.spill.compress', 'spark.broadcast.compress',
            'spark.rdd.compress'
        ]
        for col in columns_bool:
            df.loc[:, col] = df[col].apply(lambda x: 1 if x == 'True' else 0)

        onehot_encoder = OneHotEncoder(sparse=False)
        encoded_features = onehot_encoder.fit_transform(df[['spark.io.compression.codec']])
        encoded_df = pd.DataFrame(encoded_features,
                                  columns=onehot_encoder.get_feature_names_out(['spark.io.compression.codec']))
        df = pd.concat([df, encoded_df], axis=1).drop(columns=['spark.io.compression.codec'])
        return df

    def train(self, data_file):
        df = self.prepare_data(data_file)

        self.rank_model = XGBClassifierModel(model_name="xgb")
        X_train, y_train, X_test, y_test = self.rank_model.prepare_data(df, 'result')
        self.rank_model.train(X_train, y_train)
        t_s = time.time()
        y_pred = self.rank_model.predict(X_test)
        t_e = time.time()
        print(f"time: {t_e - t_s} / {len(y_pred)} = {float(t_e - t_s) / len(y_pred)}")
        self.rank_model.evaluation(X_test, y_test)

        t_s = time.time()
        self.rank_model.find_min_prediction(X_train)
        t_e = time.time()
        print(f"time: {t_e - t_s} / {len(y_pred)} = {float(t_e - t_s) / len(y_pred)}")

    def train_bo(self, data_file):
        df = self.prepare_data(data_file)
        self.pred_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
        self.pred_model.fit(df.drop(columns=["result"]), df["result"])




    def evaluate(self, X_test, y_test):
        preds = self.rank_model.predict(X_test)
        ndcg = ndcg_score([y_test], [preds])
        return ndcg

    def save_model(self, file_path):
        self.rank_model.save_model(file_path)

    def load_model(self, file_path):
        self.rank_model = xgb.XGBRanker()
        self.rank_model.load_model(file_path)


if __name__ == '__main__':
    ranker = Ranker()
    ranker.train("../data/knobs_total.csv")
    # ranker.train_bo("../data/knobs_total.csv")
    # bo()
import datetime
import time

import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from compare_model.ranker import Ranker

class BO:
    def __init__(self):
        self.param_space1 = {
            'dist_eps': (0.01, 1.0),
            'time_eps': (1, 100),
            'min_points': (10, 500),
            'max_points_per_partition': (2, 100),
            'x_bound': (0.01, 1),
            'y_bound': (0.01, 1),
            't_bound': (100, 1000)
        }
        self.param_space2 = {
            "spark.executor.cores": (1, 4),
            "spark.executor.memory": (5, 43),
            "spark.executor.instances": (3, 12),
            "spark.default.parallelism": (8, 50),
            "spark.memory.offHeap.enabled": [True, False],
            "spark.memory.offHeap.size": (10, 100),
            "spark.memory.fraction": (0.5, 1),
            "spark.memory.storageFraction": (0.5, 1),
            "spark.shuffle.file.buffer": (2, 128),
            "spark.speculation": [True, False],
            "spark.reducer.maxSizeInFlight": (2, 128),
            "spark.shuffle.sort.bypassMergeThreshold": (100, 1000),
            "spark.speculation.interval": (10, 100),
            "spark.speculation.multiplier": (1, 5),
            "spark.speculation.quantile": (0, 1),
            "spark.broadcast.blockSize": (2, 128),
            "spark.io.compression.codec": ["snappy", "lzf", "lz4"],
            "spark.io.compression.lz4.blockSize": (2, 32),
            "spark.io.compression.snappy.blockSize": (2, 32),
            "spark.kryo.referenceTracking": [True, False],
            "spark.kryoserializer.buffer.max": (8, 128),
            "spark.kryoserializer.buffer": (2, 128),
            "spark.storage.memoryMapThreshold": (50, 500),
            "spark.network.timeout": (20, 500),
            "spark.locality.wait": (1, 10),
            "spark.shuffle.compress": [True, False],
            "spark.shuffle.spill.compress": [True, False],
            "spark.broadcast.compress": [True, False],
            "spark.rdd.compress": [True, False],
            "spark.serializer": ["JavaSerializer", "KryoSerializer"]
        }
        self.col_names = list(self.param_space1.keys()) + list(self.param_space2.keys())
        self.model = None

    def search(self):

        param_space = {**self.param_space1, **self.param_space2}

        # 将参数空间转换为skopt格式
        space = []
        for param, values in param_space.items():
            if isinstance(values, tuple) and isinstance(values[0], int):
                space.append(Integer(values[0], values[1], name=param))
            elif isinstance(values, tuple) and isinstance(values[0], float):
                space.append(Real(values[0], values[1], name=param))
            elif isinstance(values, list):
                space.append(Categorical(values, name=param))

        r = Ranker()
        r.train_bo("../data/knobs_total.csv")
        self.model = r.pred_model

        def format_params(params):
            params.drop(columns=["spark.serializer"], inplace=True)

            columns_bool = [
                'spark.memory.offHeap.enabled', 'spark.speculation', 'spark.kryo.referenceTracking',
                'spark.shuffle.compress', 'spark.shuffle.spill.compress', 'spark.broadcast.compress',
                'spark.rdd.compress'
            ]
            for col in columns_bool:
                params.loc[:, col] = params[col].apply(lambda x: 1 if x == 'True' else 0)

            onehot_encoder = OneHotEncoder(sparse=False, categories=[["lz4", "lzf", "snappy"]])
            encoded_features = onehot_encoder.fit_transform(params[['spark.io.compression.codec']])
            encoded_df = pd.DataFrame(encoded_features,
                                      columns=onehot_encoder.get_feature_names_out(['spark.io.compression.codec']))
            params = pd.concat([params, encoded_df], axis=1).drop(columns=['spark.io.compression.codec'])

            return params

        def predict_execute_t(params):
            return r.pred_model.predict(format_params(params))[0]

        @use_named_args(space)
        def decorated_objective(**params):
            # 将参数传递给目标函数
            return predict_execute_t(pd.DataFrame([params]))

        t_s = time.time()
        result = gp_minimize(decorated_objective, space, n_calls=30, random_state=3)
        t_e = time.time()
        print(f"gp_minimize time: {t_e - t_s} = {float(t_e - t_s)}")

        print(f"最佳值: {result.fun}")
        print(f"最佳参数: {result.x}")



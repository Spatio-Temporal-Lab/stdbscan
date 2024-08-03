import os
import subprocess
from datetime import datetime
import random
import pandas as pd


class KnobOptimizer:
    def __init__(self):
        self.spark_version = "2.3.3"  # Example version
        self.default_conf_path = f"/usr/spark/spark-{self.spark_version}/conf/spark-defaults.conf"
        self.history_setting = (
            "spark.eventLog.enabled true\n"
            "spark.eventLog.dir hdfs://10.242.6.16:9000/user/user1/spark_history\n"
            "spark.eventLog.compress false\n"
        )

    def generate_random_params(self, n_samples, param_space):
        params_list = []
        for _ in range(n_samples):
            params = {}
            for param, space in param_space.items():
                if isinstance(space, tuple) and len(space) == 2:
                    if isinstance(space[0], float) or isinstance(space[1], float):
                        params[param] = random.uniform(space[0], space[1])
                    else:
                        params[param] = random.randint(space[0], space[1])
                elif isinstance(space, list):
                    params[param] = random.choice(space)
            params_list.append(params)
        return params_list

    def generate_spark_params(self, n_samples):
        param_space = {
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
            # "spark.serializer": ["JavaSerializer", "KryoSerializer"]
        }
        return self.generate_random_params(n_samples, param_space), list(param_space.keys())

    def generate_stdbscan_params(self, n_samples):
        param_space = {
            'dist_eps': (0.01, 1.0),
            'time_eps': (1, 100),
            'min_points': (10, 500),
            'max_points_per_partition': (2, 100),
            'x_bound': (0.01, 1),
            'y_bound': (0.01, 1),
            't_bound': (100, 1000)
        }
        return self.generate_random_params(n_samples, param_space), list(param_space.keys())

    def convert_units(self, conf):
        units = {
            "spark.executor.memory": "g",
            "spark.memory.offHeap.size": "m",
            "spark.shuffle.file.buffer": "k",
            "spark.reducer.maxSizeInFlight": "m",
            "spark.speculation.interval": "m",
            "spark.broadcast.blockSize": "m",
            "spark.io.compression.lz4.blockSize": "m",
            "spark.io.compression.snappy.blockSize": "m",
            "spark.kryoserializer.buffer.max": "m"
        }
        for key, unit in units.items():
            if key in conf:
                conf[key] = str(conf[key]) + unit
        return conf

    def update_spark_conf(self, conf):
        with open(self.default_conf_path, 'w') as file:
            file.write(self.history_setting)
            for key, value in conf.items():
                file.write(f"{key} {value}\n")
        self.restart_spark()

    def restart_spark(self):
        cmd_stop = f"/usr/spark/spark-{self.spark_version}/sbin/stop-all.sh"
        cmd_start = f"/usr/spark/spark-{self.spark_version}/sbin/start-all.sh"
        subprocess.run(cmd_stop, shell=True)
        subprocess.run(cmd_start, shell=True)

    def run_kl(self, knobs):
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dist_eps = knobs["dist_eps"]
        time_eps = knobs["time_eps"]
        min_points = knobs["min_points"]
        max_points_per_partition = knobs["max_points_per_partition"]
        x_bound = knobs["x_bound"]
        y_bound = knobs["y_bound"]
        t_bound = knobs["t_bound"]
        cmd = (
            f"spark-submit --class org.apache.spark.Scala.DBScan3DDistributed.DBScan3DDistributedTest "
            f"../jar/newyork_kl_sample.jar misa/data/newyork result//{current_date_time} "
            f"{dist_eps} {time_eps} {min_points} {max_points_per_partition} {x_bound} {y_bound} {t_bound}"
        )
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        total = result.stdout.split("\n")[-16:-2]
        for cost in total:
            print(cost)
        return result.returncode, total[-1].split(" ")[1]

    def optimize_knobs(self, n_samples=200):
        params_stdbscan, keys_stdbscan = self.generate_stdbscan_params(n_samples)
        params_spark, keys_spark = self.generate_spark_params(n_samples)

        df = pd.DataFrame(columns=keys_stdbscan + keys_spark + ['result'])
        file_path = "../result/knobs/knobs_total.csv"
        if not os.path.exists(file_path):
            df.to_csv(file_path, sep=',', index=False)

        for i, (pdb, psp) in enumerate(zip(params_stdbscan, params_spark)):
            print(f"knobs setting {i + 1}: {pdb} {psp}")
            self.update_spark_conf(self.convert_units(psp))
            returncode, total = self.run_kl(pdb)
            params = {**pdb, **psp}
            params['result'] = total
            df.loc[len(df)] = params
            print(df)

            df_new = pd.DataFrame([params])
            df_new.to_csv(file_path, sep=',', mode='a', header=False, index=False)

        # Optionally, return or save the final dataframe
        return df


if __name__ == '__main__':
    optimizer = KnobOptimizer()
    optimizer.optimize_knobs()

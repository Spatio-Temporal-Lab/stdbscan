from itertools import combinations

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class XGBRankerModel:
    def __init__(self, params=None, num_boost_round=100):
        """
        初始化 XGBRankerModel 类
        :param params: XGBoost 参数字典
        :param num_boost_round: 迭代次数
        """
        if params is None:
            params = {
                'objective': 'rank:pairwise',
                'learning_rate': 0.1,
                'gamma': 1.0,
                'min_child_weight': 0.1,
                'max_depth': 5,
                'n_estimators': 100
            }
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = None

    def prepare_data(self, df, target_column):
        """
        准备数据并进行训练和测试集分割
        :param df: 数据框
        :param feature_columns: 特征列名列表
        :param target_column: 目标列名
        :return: 训练集和测试集的特征、目标信息
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # 使用 sklearn 的 train_test_split 进行数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    def train(self, X_train, y_train):
        """
        训练 XGBRanker 模型
        :param X_train: 训练集特征
        :param y_train: 训练集目标
        """
        # 转换为 DMatrix 格式
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        train_dmatrix.set_group([len(y_train)])  # 只有一个组，长度就是训练集的样本数

        # 训练模型
        self.model = xgb.train(self.params, train_dmatrix, num_boost_round=self.num_boost_round)

    def predict(self, X_test):
        """
        使用训练好的模型进行预测
        :param X_test: 测试集特征
        :return: 预测值
        """
        test_dmatrix = xgb.DMatrix(X_test)
        test_dmatrix.set_group([len(X_test)])  # 只有一个组，长度就是测试集的样本数

        # 预测
        y_pred = self.model.predict(test_dmatrix)
        return y_pred

    def find_min_prediction(self, y_pred):
        """
        找出预测值最小的项
        :param y_pred: 预测值
        :return: 最小项的索引和预测值
        """
        min_index = y_pred.argmin()
        min_value = y_pred[min_index]
        return min_index, min_value


class XGBClassifierModel:
    def __init__(self, num_boost_round=100, model_name="xgb"):
        self.num_boost_round = num_boost_round
        self.model = None
        self.model_name = model_name
        self.data = None

    def prepare_data(self, df, target_column):
        """
        准备数据并进行训练和测试集分割
        :param df: 数据框
        :param feature_columns: 特征列名列表
        :param target_column: 目标列名
        :return: 训练集和测试集的特征、目标信息
        """
        self.data = df
        combinations_list = list(combinations(df.index, 2))
        combined_data = []
        for i, j in combinations_list:
            row1 = df.iloc[i]
            row2 = df.iloc[j]
            combined_row = list(row1) + list(row2)
            label = int(row1['result'] > row2['result'])
            combined_row.append(label)
            combined_data.append(combined_row)

        new_columns = [f"{col}_1" for col in df.columns] + [f"{col}_2" for col in df.columns] + [target_column]

        combined_df = pd.DataFrame(combined_data, columns=new_columns)
        combined_df.drop(columns=[f"{target_column}_1", f"{target_column}_2"], inplace=True)

        X = combined_df.drop(target_column, axis=1)
        y = combined_df[target_column]

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    def train(self, X_train, y_train):
        """
        训练 XGBRanker 模型
        :param X_train: 训练集特征
        :param y_train: 训练集目标
        """
        if self.model_name == "xgb":
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'learning_rate': 0.1,
                'nthread': -1
            }
            # 转换为 DMatrix 格式
            train_dmatrix = xgb.DMatrix(X_train, label=y_train)
            # 训练模型
            self.model = xgb.train(params, train_dmatrix, num_boost_round=self.num_boost_round)
        elif self.model_name == "lr":
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X_train, y_train)
        elif self.model_name == "svm":
            self.model = SVC()
            self.model.fit(X_train, y_train)
        elif self.model_name == "dt":
            self.model = DecisionTreeClassifier()
            self.model.fit(X_train, y_train)
        elif self.model_name == "rf":
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
        elif self.model_name == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5)
            self.model.fit(X_train, y_train)
        elif self.model_name == "nb":
            self.model = GaussianNB()
            self.model.fit(X_train, y_train)
        elif self.model_name == "nn":
            self.model = MLPClassifier(max_iter=300)
            self.model.fit(X_train, y_train)
        elif self.model_name == "lgb":
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            # 训练模型
            self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict(self, X_test, y_test = None):
        """
        使用训练好的模型进行预测
        :param X_test: 测试集特征
        :return: 预测值
        """
        if self.model_name == "xgb":
            test = xgb.DMatrix(X_test)
            y_pred_prob = self.model.predict(test)
            y_pred = (y_pred_prob > 0.5).astype(int)
        # elif self.model_name == "lgb":
        #     test = lgb.Dataset(X_test, label=y_test)
        else:
            test = X_test
            # 预测
            y_pred = self.model.predict(test)

        return y_pred

    def evaluation(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

    def find_min_prediction_d(self):
        df = self.data.drop(columns=["result"])
        combinations_list = list(combinations(df.index, 2))
        result = pd.DataFrame({"wins": [0] * len(df)})
        for i, j in combinations_list:
            row1 = df.iloc[i]
            row2 = df.iloc[j]
            combined_row = list(row1) + list(row2)
            if self.model_name == "xgb":
                new_columns = [f"{col}_1" for col in df.columns] + [f"{col}_2" for col in df.columns]
                combined_df = pd.DataFrame([combined_row], columns=new_columns)
                test = xgb.DMatrix(combined_df)
                y_pred_prob = self.model.predict(test)
                compare_result = (y_pred_prob > 0.5).astype(int)
            else:
                compare_result = self.model.predict([combined_row])[0]

            if compare_result == 0:
                result.at[i, "wins"] += 1
            else:
                result.at[j, "wins"] += 1
        min_index = result['wins'].idxmax()
        min_record = df.loc[min_index]

        print(min_index, min_record)
        return min_index, min_record

    def find_min_prediction(self, df):
        result = self.predict(df)
        # print(len(result))  #15288
        n = 1000
        losses = [0] * n

        index = 0
        for i in range(n):
            for j in range(i + 1, n):
                if index >= len(result):
                    break
                if result[index] == 0:
                    losses[i] += 1
                else:
                    losses[j] += 1
                index += 1
        min_index = losses.index(max(losses))

        return min_index, df.loc[min_index]


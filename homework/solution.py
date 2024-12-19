import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import gzip
import json, os
np.set_printoptions(legacy='1.25')
from sklearn.metrics import (
    r2_score, mean_squared_error, median_absolute_error, mean_absolute_error)

class Lab05:
    def __init__(self) -> None:
        self.time = time.time()
        self.files_path = 'files/'
        self.columnas_categoricas = ['Fuel_Type','Selling_type','Transmission']

        self.param_grid = {
            'selectk__k': range(1,12),
        }
        print(self.param_grid)

    def main(self):
        df_train = self.read_dataset('input/train_data.csv.zip')
        df_test = self.read_dataset('input/test_data.csv.zip')
        df_train = self.clean_dataset(df_train)
        df_test = self.clean_dataset(df_test)
        X_train,  y_train = self.train_test_split(df_train)
        X_test,  y_test = self.train_test_split(df_test)
        self.columnas_no_categoricas = list(set(X_train.columns.values) - set(self.columnas_categoricas))
        pipeline = self.make_pipeline(LinearRegression(n_jobs=-1, fit_intercept=True))
        estimator = self.make_grid_search(pipeline, 'neg_mean_absolute_error')#, cv=StratifiedKFold(n_splits=10,shuffle=False))
        estimator = estimator.fit(X_train, y_train)
        self.save_estimator(estimator)#self.save_model_if_best(estimator, X_train, y_train)
        print(estimator.best_params_)
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test) 
        metrics_train = self.eval_metrics('train', y_train, y_train_pred)
        metrics_test = self.eval_metrics('test', y_test, y_test_pred)
        self.save_metrics(metrics_train, metrics_test)
        print(f'Minutos: {(time.time() - self.time)/60}')



    def read_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(self.files_path + path)
        return df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Age'] = 2021 - df['Year']
        df.drop(['Year','Car_Name'], axis=1, inplace=True)
        return df
    
    def train_test_split(self, df):
        return df.drop('Present_Price', axis=1), df['Present_Price']
    
    def make_pipeline(self, estimator):
        transformer = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(dtype='int'), self.columnas_categoricas),
            ],
            remainder='passthrough'
        )

        pipeline = Pipeline(
            steps=[
                ('transformer', transformer),
                ('scaler', MinMaxScaler()),
                ('selectk', SelectKBest(score_func=f_regression, k='all')),
                ('linear_model', estimator)
            ]
        )
        return pipeline
        
    def make_grid_search(self, estimator, scoring, cv=10):
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        return grid_search

    def save_model_if_best(self, estimator, X, y):
        best_estimator = self.load_model()
        if best_estimator:
            saved_error = mean_absolute_error(y, best_estimator.predict(X))
            current_error = mean_absolute_error(y, estimator.predict(X))

            if current_error < saved_error:
                self.save_estimator(estimator)
            else:
                estimator = best_estimator
        else:
            self.save_estimator(estimator)
        return estimator


    def save_estimator(self, estimator):
        with gzip.open(self.files_path + 'models/model.pkl.gz', 'wb') as file:
            pickle.dump(estimator, file)

    def load_model(self):
        try:
            with gzip.open(self.files_path + "models/model.pkl.gz", "rb") as file:
                estimator = pickle.load(file)
            return estimator
        except Exception as E:
            print(E)
            return None
    
    def eval_metrics(self, dataset,y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mad = median_absolute_error(y_true, y_pred)
        return {"type": "metrics","dataset": dataset, "r2": r2, "mse": mse, "mad": mad} 
    
    
    def save_metrics(self, metrics_train, metrics_test):
        os.remove(self.files_path + 'output/metrics.json')
        with open(self.files_path + 'output/metrics.json', mode='w') as file:
            file.write(json.dumps(metrics_train)+"\n")
            file.write(json.dumps(metrics_test)+"\n")


if __name__=='__main__':
    obj = Lab05()
    obj.main()
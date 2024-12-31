import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as sf
import warnings
warnings.filterwarnings('ignore')


class Regression:

    def load_data(self):
        try:
            df = pd.read_csv('./data/insurance.csv')
            print(df.head())
            return df
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def explore_data(self, df):
        try:
            df.isnull().sum()  #check if there is any null in data
            plt.figure()
            sns.histplot(x='age', data=df, bins=20, kde=True)
            plt.show()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def preprocessing_data(self, df):

        try:
            encoder = LabelEncoder()
            df['sex'] = encoder.fit_transform(df['sex'])
            df['smoker'] = encoder.fit_transform(df['smoker'])
            df['region'] = encoder.fit_transform(df['region'])
            return df
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def correlation_vars(self, df):
        sns.heatmap(df.corr(), annot=True, cmap='Paired', fmt='.2f')
        plt.show()

    def ols_analysis(self, df):

        try:
            model = sf.ols('charges ~ age + sex + bmi + children + smoker + region', data=df)
            results = (model.fit()).summary()
            print(results)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def features_eng(self, df):

        try:
            # feature target selection for training
            x = df.drop('charges', axis=1)
            y = df['charges']
            scaler = StandardScaler()
            for col in x.columns:
                x[col] = scaler.fit_transform(x[[col]])
            return x, y
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


if __name__ == '__main__':
    lr = Regression()
    DataDf = lr.load_data()
    lr.explore_data(DataDf)
    DataDf = lr.preprocessing_data(DataDf)
    lr.correlation_vars(DataDf)
    lr.ols_analysis(DataDf)

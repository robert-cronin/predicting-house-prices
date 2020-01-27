import pandas as pd
import numpy as np
import operator

class PredictHousePrices():
    def __init__(self):
        self.train_data = pd.read_csv("data/train.csv")
        self.test_data = pd.read_csv("data/test.csv")

    # def data_prep(self):
    #     self.test_data.dropna(inplace=True)

    def top_correlated_columns(self):
        correlations = {}
        for col in self.test_data:
            if col != "SalePrice" and col != "Id" and (self.train_data[col].dtype == np.float64 or self.train_data[col].dtype == np.int64):
                correlations[col] = abs(self.calculate_correlation(col))
        return dict(sorted(correlations.items(), key=operator.itemgetter(1), reverse=True)[:10])


    def calculate_correlation(self, col):
        return self.train_data[col].corr(self.train_data["SalePrice"])



if __name__ == "__main__":
    sol = PredictHousePrices()
    print("Top 5 correlated numeric columns are: ")
    print(sol.top_correlated_columns())
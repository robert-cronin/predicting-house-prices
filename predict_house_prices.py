import statsmodels.api as stats
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

class PredictHousePrices():
    def __init__(self):
        self.train_data = pd.read_csv("data/train.csv")
        self.learn_data = self.train_data[0:int(self.train_data.shape[0]*0.8)]
        self.validate_data = self.train_data[int(self.train_data.shape[0]*0.8):-1]

        self.test_data = pd.read_csv("data/test.csv")

    # def data_prep(self):
    #     self.test_data.dropna(inplace=True)

    def numeric_column_correlations(self):
        correlations = {}
        for col in self.test_data:
            if col != "SalePrice" and col != "Id" and (self.train_data[col].dtype == np.float64 or self.train_data[col].dtype == np.int64):
                correlations[col] = abs(self.calculate_correlation(col))
        return dict(sorted(correlations.items(), key=operator.itemgetter(1), reverse=True))

    def top_correlated_columns(self):
        correlations = self.numeric_column_correlations()
        return dict(sorted(correlations.items(), key=operator.itemgetter(1), reverse=True)[:10])

    def show_training_data_correlations(self):
        numeric_columns = self.train_data.select_dtypes([np.float64, np.int64, np.float64])
        f = plt.figure(figsize=(15, 19))
        plt.matshow(numeric_columns.corr(), fignum=f.number)
        plt.xticks(range(numeric_columns.shape[1]), numeric_columns.columns, fontsize=14, rotation=90)
        plt.yticks(range(numeric_columns.shape[1]), numeric_columns.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('House Prices Corr Matrix', fontsize=30, y=1.2)
        plt.show()

    def calculate_correlation(self, col):
        return self.train_data[col].corr(self.train_data["SalePrice"])

    def build_linreg_model(self, col):
        model = stats.OLS(self.learn_data["SalePrice"], self.learn_data[col]).fit()
        return model


    def calc_rmse(self, predictions):
        return np.sqrt(((predictions - self.validate_data["SalePrice"]) ** 2).mean())


if __name__ == "__main__":
    sol = PredictHousePrices()
    # Show correlation graph
    sol.show_training_data_correlations()
    print("Top 5 correlated numeric columns are: ")
    corr_cols = sol.top_correlated_columns()
    for col, corr in corr_cols.items():
        print(col+": ", corr)
    for col in corr_cols:
        model = sol.build_linreg_model(col)

        predictions = model.predict(sol.validate_data[col])

        print("RMSE for Predictions for "+col+": "+str(sol.calc_rmse(predictions)))


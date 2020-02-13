import statsmodels.api as stats
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

class PredictHousePrices():
    def __init__(self):
        missing_values = ["n/a", "na", "--", "nan"]
        self.train_data = pd.read_csv("data/train.csv", na_values=missing_values)
        print(self.train_data[self.train_data.isnull().any(axis=1)])
        self.learn_data = self.train_data[0:int(self.train_data.shape[0]*0.8)]
        self.validate_data = self.train_data[int(self.train_data.shape[0]*0.8):-1]

        self.test_data = pd.read_csv("data/test.csv")

    def data_prep(self):
        # Remove duplicated rows
        print("Shape before drop_duplicates: ", self.train_data.shape)
        # self.train_data.drop_duplicates(subset="First Name", keep=False, inplace=True)
        self.train_data.drop_duplicates(keep=False, inplace=True)
        print("Shape before dropna: ", self.train_data.shape)
        self.train_data.dropna(axis=1, thresh=50, inplace=True)
        print("Shape after dropna: ", self.train_data.shape)

        print("")
        # Fill mean for float columns
        for col in self.train_data.select_dtypes(include=['float64']):
            mean = self.train_data[col].mean()
            self.train_data[col] = self.train_data[col].fillna(mean)
        # Fill mode for string and int columns
        string_int_columns = self.train_data.applymap(lambda x: isinstance(x, str)).all(0)
        for col in self.train_data[self.train_data.columns[string_int_columns]]:
            mode = self.train_data[col].mode()
            self.train_data[col] = self.train_data[col].fillna(mode)

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
    sol.data_prep()
    # Show correlation graph
    # sol.show_training_data_correlations()
    # print("Top 5 correlated numeric columns are: ")
    # corr_cols = sol.top_correlated_columns()
    # for col, corr in corr_cols.items():
    #     print(col+": ", corr)
    # for col in corr_cols:
    #     model = sol.build_linreg_model(col)
    #
    #     predictions = model.predict(sol.validate_data[col])
    #
    #     print("RMSE for Predictions for "+col+": "+str(sol.calc_rmse(predictions)))


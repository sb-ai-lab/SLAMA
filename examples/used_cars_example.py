# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""

import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


# load and prepare data
# TODO: put a correct path for used_cars dataset
cols = list(pd.read_csv("/opt/small_used_cars_data_cleaned.csv", nrows =1))

data = pd.read_csv("/opt/small_used_cars_data_cleaned.csv", usecols = [i for i in cols if i not in ["dealer_zip", "description"]])
#data = pd.read_csv("./data/used_cars_data.csv", usecols = [i for i in cols if i in ["daysonmarket", "front_legroom", "fuel_tank_volume", "height", "price"]])
#data = pd.read_csv("./data/used_cars_data.csv", usecols = [i for i in cols if i in ["horsepower", "latitude", "length", "listed_date", "listing_id", "longitude",  "price"]])
#data = pd.read_csv("./data/used_cars_data.csv", usecols = [i for i in cols if i in ["maximum_seating", "mileage", "owner_count", "power", "savings_amount", "seller_rating",  "price"]])
#data = pd.read_csv("./data/used_cars_data.csv", usecols = [i for i in cols if i in ["vin", "year", "width", "trimId", "sp_id",  "price"]])
#data = pd.read_csv("./data/used_cars_data.csv", usecols = [i for i in cols if i in ["back_legroom", "year", "width", "trimId", "sp_id",  "price"]])
data = pd.concat([data, data], ignore_index=True)
#data = pd.concat([data, data], ignore_index=True)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# run automl
if __name__ == "__main__":
    task = Task("reg")

    automl = TabularAutoML(task=task, general_params={"use_algos": ["lgb", "linear_l2"]})

    oof_predictions = automl.fit_predict(train_data, roles={"target": "price"})
    te_pred = automl.predict(test_data)

    # calculate scores
    # TODO: replace with mse
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    #print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
    #print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")

    # data.to_csv("./data/derivative_datasets/2x_dataset.csv")
    print('ready')


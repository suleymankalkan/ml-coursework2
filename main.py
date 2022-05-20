import pandas as pandas
import numpy as numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

# Columns to drop that holds ineffective or unnecessary data for training
INEFFECTIVE_COLUMNS = ["id", "atmo_opacity", "wind_speed"]
# Target columns to predict after training
TARGET_COLUMNS = ["min_temp", "max_temp"]
# Columns that hold data for training
training_columns = [
    "terrestrial_date",
    "sol",
    "ls",
    "month",
    "pressure",
]
# Test data percentage to split from the whole dataset
TEST_PERCENTAGE = 20
# Integer or float columns that is going to be normalized
normalized_columns = [
    "sol",
    "ls",
    "pressure"
]

# Import Dataset
dataset = pandas.read_csv('mars-weather.csv')
# Preprocessing: Change terrestrial_date from string to datetime for optimization
dataset['terrestrial_date'] = pandas.DatetimeIndex(dataset['terrestrial_date']).astype(numpy.int64) / 1000000
# Preprocessing: Change month columns from ex: "Month 5" to 5
dataset['month'] = dataset['month'].str.split(' ').str[1]
# Preprocessing: Drop ineffective columns
dataset = dataset.drop(columns=INEFFECTIVE_COLUMNS)
# Preprocessing: Drop rows that have NaN value
dataset.dropna(inplace=True)
# Preprocessing: Normalize columns for better performance in algorithms except date
normalizer = MinMaxScaler()
dataset[normalized_columns] = normalizer.fit_transform(dataset[normalized_columns])

# Split the data as target and training data
X = dataset.drop(columns=TARGET_COLUMNS)
Y = dataset.drop(columns=training_columns)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_PERCENTAGE / 100)

# Training models list
TRAINING_MODELS = {
    "Neural Network": MLPRegressor(random_state=1, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Linear Regressor": LinearRegression()
}


# Train and test models then print the metrics
def train_and_test_model(model_name, model):
    model.fit(X, Y)
    prediction = model.predict(X_test)
    mse = mean_squared_error(Y_test, prediction)
    cv_score = cross_val_score(model, X, Y, cv=10)

    print("\nAlgorithm: " + model_name)
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % mse ** (1 / 2.0))
    print("Cross validation mean: ", cv_score.mean())


# Execute train_and_test_model function for every model in the model list
for name, model in TRAINING_MODELS.items():
    train_and_test_model(name, model)


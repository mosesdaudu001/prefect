import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print('Program starting...')

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

mlflow.set_tracking_uri('sqlite:///backend.db')
mlflow.set_experiment('moses-nyc-taxi-experiment')

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():

        mlflow.set_tag("developer", "Moses Daudu")

        mlflow.sklearn.autolog()

        mlflow.log_param('train_data', os.path.join(data_path, 'train.pkl'))
        mlflow.log_param('val_data', os.path.join(data_path, 'val.pkl'))

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)
        print('The root meansquared error is:', rmse)
        print('Program ended')


if __name__ == '__main__':
    run_train()

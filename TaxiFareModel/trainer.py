from bleach import clean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, DistanceTransformer2
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, ElasticNet
from sklearn.compose import ColumnTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import mlflow
from memoized_property import memoized_property
import joblib


class Trainer():
    
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y        
        self.model = kwargs.get("model", LinearRegression())

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer2()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (f"{self.model}".rstrip("()"), self.model)
        ])
        self.pipeline = pipe

    def run(self):
        '''returns a trained pipelined model'''
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        self.experiment_name = "[FR] [Bdx] [HumbertMonnot] LinearRegresion v2"
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self, name):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, name)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # build pipeline
    for estimator in [LinearRegression(), Lasso(), SGDRegressor(), ElasticNet()]:
        train = Trainer(X_train, y_train)

        # train the pipeline
        train.run()
        #train.save_model(f"{estimator}".rstrip("()"))
        train.evaluate(X_val, y_val)
        
        # evaluate the pipeline
        #train.mlflow_log_param("model", estimator)
        #train.mlflow_log_metric("RMSE", train.evaluate(X_val, y_val))
        
    experiment_id = train.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
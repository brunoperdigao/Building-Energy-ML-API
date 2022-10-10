from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Json
import pandas as pd
import pickle
import bz2file as bz2

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler

future_covariates_path = "./models/future_covariates.pkl"
future_series = TimeSeries.from_pickle(future_covariates_path)
data = bz2.BZ2File('./models/ts_model.pbz2', 'rb')
model = pickle.load(data)
print(model)

class Parameters(BaseModel):
    n_days_to_predict: int
    df_in: str
    
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running."}

@app.get("/future_covariates/")
def get_future_covariates():
    return FileResponse(future_covariates_path)

@app.post("/prediction/")
def get_prediction(item: Parameters):
    df = pd.read_json(item.df_in)
    series = TimeSeries.from_dataframe(df,
                                       time_col='timestamp',
                                       fill_missing_dates=True)
    scaler = Scaler(scaler=MinMaxScaler(feature_range=(0.005, 1)))
    series_scaled = scaler.fit_transform(series)
    pred = model.predict(9,
                         series=series_scaled,
                         future_covariates=future_series)
    #
    scaler_inverse = Scaler()
    scaler_inverse = scaler_inverse.fit(series)
    # series_inverted = scaler_inverse.inverse_transform(series_scaled)
    # series_inverted[-90:].plot(label='train')
    pred_inverted = scaler_inverse.inverse_transform(pred)
    # pred_inverted.plot(label='prediction')
    return (pred_inverted.pd_dataframe().to_json())
    

### Import model
### Create a function to get future_covariates
### Future covariates will be passed to dashboard and be updated there
### Create a function to with parameters (future_covariates and n_days_to_predict)
### Return a JSON or Dataframe?

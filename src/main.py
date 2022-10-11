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

data = bz2.BZ2File('./models/ts_model.pbz2', 'rb')
model = pickle.load(data)

class Parameters(BaseModel):
    n_days_to_predict: int
    json_fut_cov: str
    json_building: str
    
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running."}


@app.post("/prediction/")
def get_prediction(item: Parameters):
    
    #####
    df_fut_cov = pd.read_json(item.json_fut_cov)
    future_series = TimeSeries.from_dataframe(
                                    df_fut_cov,
                                    time_col='timestamp',
                                    fill_missing_dates=True,
                                              )
    scaler_fut_cov = Scaler(scaler=MinMaxScaler(feature_range=(0.005, 1)))
    future_series_scaled = scaler_fut_cov.fit_transform(future_series)
    

    #####
    df = pd.read_json(item.json_building)
    series = TimeSeries.from_dataframe(df,
                                       time_col='timestamp',
                                       fill_missing_dates=True)
    scaler = Scaler(scaler=MinMaxScaler(feature_range=(0.005, 1)))
    series_scaled = scaler.fit_transform(series)
    print(series_scaled)
    pred = model.predict(9,
                         series=series_scaled,
                         future_covariates=future_series_scaled)
    #
    scaler_inverse = Scaler()
    scaler_inverse = scaler_inverse.fit(series)
    # series_inverted = scaler_inverse.inverse_transform(series_scaled)
    # series_inverted[-90:].plot(label='train')
    pred_inverted = scaler_inverse.inverse_transform(pred)
    # pred_inverted.plot(label='prediction')
    return (pred_inverted.pd_dataframe().to_json())
    


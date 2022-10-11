import pandas as pd
import requests
import pickle
from darts import TimeSeries

from datetime import date

TODAY = str(date.today())


###########

params = {
    "latitude":"60.19",
    "longitude":"24.94",
    "start_date":"2020-01-01",
    "end_date":TODAY,
    "hourly": ["temperature_2m", "relativehumidity_2m"],
    "timezone": "auto",
    }
endpoint = "https://archive-api.open-meteo.com/v1/era5"
response = requests.get(endpoint, params=params)
response.json()['hourly'].keys()

response_dict = response.json()['hourly']
df_hist_weather = pd.DataFrame.from_dict(response_dict)
df_hist_weather['time'] = pd.to_datetime(df_hist_weather['time'])
df_hist_weather.rename(columns={'time': 'timestamp'}, inplace=True)
df_hist_weather.set_index(['timestamp'], inplace=True)
df_hist_weather = df_hist_weather.resample('D').mean()


# %% colab={"base_uri": "https://localhost:8080/", "height": 238} id="kA8Mkj6bC7fc" outputId="1c3a57b3-b80e-478a-da11-255daeb8be72"
df_hist_weather.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} id="kKh4U0lvEgrD" outputId="fefcdce2-5756-4d7d-b33e-0895834d83c7"
df_hist_weather = df_hist_weather[df_hist_weather['temperature_2m'].notna()]
df_hist_weather.tail(10)

# %% id="S6gWQ2ECDKmP"
params = {
    "latitude":"60.19",
    "longitude":"24.94",
    "hourly": ["temperature_2m", "relativehumidity_2m"],
    "past_days": 15,
    "timezone": "auto",
    }
endpoint = "https://api.open-meteo.com/v1/forecast"
response = requests.get(endpoint, params=params)
response.json()['hourly'].keys()


response_dict = response.json()['hourly']
df_fore_weather = pd.DataFrame.from_dict(response_dict)
df_fore_weather['time'] = pd.to_datetime(df_fore_weather['time'])
df_fore_weather.rename(columns={'time': 'timestamp'}, inplace=True)
df_fore_weather.set_index(['timestamp'], inplace=True)
df_fore_weather = df_fore_weather.resample('D').mean()



# %% colab={"base_uri": "https://localhost:8080/", "height": 771} id="6DUXgR1_DcAW" outputId="badf04b5-d585-4d1f-dbf9-5cc951d8c85c"
df_fore_weather

# %% colab={"base_uri": "https://localhost:8080/"} id="KSBPurt4DvML" outputId="cc8e5886-0bc9-4720-9025-d02726526ebc"
temp_hist = df_hist_weather['temperature_2m']
temp_fore = df_fore_weather['temperature_2m']
temp_all = temp_hist.combine_first(temp_fore)


# %% colab={"base_uri": "https://localhost:8080/"} id="obM6kng5D_p3" outputId="891506ec-751f-406e-f721-d46dd3b7662f"
humidity_hist = df_hist_weather['relativehumidity_2m']
humidity_fore = df_fore_weather['relativehumidity_2m']
humidity_all = humidity_hist.combine_first(humidity_fore)

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="gBGsNvl4K5CR" outputId="035c0123-7846-4a6b-fc59-1b3ef5c30064"
df_all_weather = pd.concat([temp_all, humidity_all], axis=1)
df_all_weather

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="y_8NZMoXNGt4" outputId="b9750ec6-f886-447f-d2b6-5f0d8327dde5"
df_all_weather['weekend'] = (df_all_weather.index.dayofweek >= 5).astype(int)
df_all_weather['month'] = df_all_weather.index.month

df_all_weather = df_all_weather.reset_index()

df_all_weather = df_all_weather.to_json()


###########

endpoint_energy = "https://helsinki-openapi.nuuka.cloud/api/v1/EnergyData/Daily/ListByProperty"

params_energy = {
        'Record': 'propertyCode',
        'SearchString': '091-004-0001-0012',
        'ReportingGroup': 'Electricity',
        'StartTime': '2020-01-01',
        'EndTime': TODAY,
        }
response_energy = requests.get(endpoint_energy, params=params_energy)
print(response_energy.status_code)

new_building = pd.DataFrame.from_dict(response_energy.json())
new_building['timestamp'] = pd.to_datetime(new_building['timestamp'])
column_name = f'Building {0}'
new_building = new_building.rename(columns={'value': column_name})
new_building = new_building.drop(['reportingGroup', 'locationName', 'unit'], axis=1)
new_building = new_building.to_json()

params = {
        "n_days_to_predict": 9,
        "json_fut_cov": df_all_weather,
        "json_building": new_building
        }
endpoint2 = 'http://127.0.0.1:8000/prediction/'
response = requests.post(endpoint2, json=params)
if response:
    print(response.text)
else:
    print('vazio')

import pandas as pd
import requests
import pickle
from darts import TimeSeries

from datetime import date

TODAY = str(date.today())


endpoint1 = 'http://127.0.0.1:8000/future_covariates/'
response = requests.get(endpoint1, stream=True)
ts = pickle.load(response.raw)
print(ts.columns)



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
print(new_building.head(5))
new_building = new_building.to_json()

params = {
        "n_days_to_predict": 9,
        "df_in": new_building
        }
endpoint2 = 'http://127.0.0.1:8000/prediction/'
response = requests.post(endpoint2, json=params)
if response:
    print(response.text)
else:
    print('vazio')

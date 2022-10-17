# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="BbFWRMJ-lT4n"
# # Treinamento de modelo de Machine Learning para análise de série temporal e previsão de consumo energético em edificações
#
# Este notebook demonstra como foi feito o treinamento de um modelo para utilização neste [Dashboard](https://building-energy-dashboard.onrender.com)  sobre consumo energético em edificações. Para este processo utilizaremos algumas bibliotecas básicas de ciência de dados, com destaque para a biblioteca Darts, especializada em Machine Learning para séries temporais. Iniciamos com as devidas instalações e importações.

# %% colab={"base_uri": "https://localhost:8080/"} id="jH1R1LT7Rc8-" outputId="4b449395-e674-43a2-de9d-08017f394140"
# Necessary for Google Colab
# !pip install darts --quiet
# !pip install matplotlib==3.1.3 --quiet
# !pip install bz2file --quiet

# %% id="aAOdqfZpj0_w"
import pandas as pd
import numpy as np
import pickle
import bz2file as bz2
import requests
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")


from datetime import date
from typing import Tuple

TODAY = str(date.today())

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics.metrics import mape, mase, smape, r2_score, rmse

from sklearn.preprocessing import MinMaxScaler


# %% [markdown] id="alkJMxWFmGK_"
# ## Dados principais - Edificações
#
# Os dados utilizados nesse processo são coletados de uma API fornecida pela prefeitura de Helsinki, na Finlândia. Para mais detalhes sobre esse processo, visite a explicação completa sobre o [Dashboard](https://github.com/brunoperdigao/Building-Energy-Dashboard#readme).
# Inicialmente vamos coletar a lista de edificações que se enquadram no tipo de uso "escritório". E para cada edifício, pegamos seu `propertyCode` que vamos usar posteriormente para coletar seus dados de consumo energético. Nesse processo, descartamos alguns códigos que estão vazios ou fora do padrão.

# %% [markdown] id="OMZmqy4NvjBy"
# ## Main data - Buildings
# The data used in this process are collect from a API provided by Helsink mucinipality, Finland. To get more details about this, visit the [dashboard page](https://github.com/brunoperdigao/Building-Energy-Dashboard#readme)
# First we are going to collect the list of buildings that match the "office" type. For each building, we get its `propertyCode` that we are going to use latter to colect the data for energy consumption. In this process, we discharge a few codes that are empty or not following the standars.

# %% id="BnDKROQGj-F7"
endpoint_offices = "https://helsinki-openapi.nuuka.cloud/api/v1.0/Property/Search"
params_offices = {
    "SearchString": "office",
    "SearchFromRecord": "BuildingType",
}
response_offices = requests.get(endpoint_offices, params=params_offices)

# Cleaning the nested keys
offices_list = response_offices.json()
to_exclude = ["reportingGroups", "buildings"]
new_list = []
for item in offices_list:
    new_item = {key: item[key] for key in item.keys() if key not in to_exclude}
    new_list.append(new_item)
df = pd.DataFrame.from_records(new_list)

propertyCodes = list(df["propertyCode"])

for item in propertyCodes:
    if (len(item) < 17) or (len(item) > 17):
        propertyCodes.remove(item)

len(propertyCodes)

# %% [markdown] id="3zUs30JJnNTQ"
# Agora com os `propertyCode` coletados, vamos fazer uma busca pelos dados de cada edificação. No código abaixo montamos um Dataframe com os dados de todas as edificações. Para evitar erros e dados incompletos, primeiro testamos se a resposta da API é positiva (200), se os dados estão vazios ou se contém dados duplicados. Esses critérios foram utilizados pela observação desses problemas ao longo do processo. Dos 163 `propertyCodes` selecionados, apenas 70 foram bem sucedidos.
# ///
# Now with the `propertyCode` collected, let's do a search for the data of each building. In the code below we set up a Dataframe with data from all buildings. To avoid errors and incomplete data, we first test if the API response is positive (200), if the data is empty or if it contains duplicate data. These criteria were used by observing these problems throughout the process. Of the 163 `propertyCodes` selected, only 70 were successful.

# %% colab={"base_uri": "https://localhost:8080/"} id="-DyBra6GkxyJ" outputId="e9b6d469-9ad2-4d54-b25e-be2ac4a61e54"
endpoint_energy = (
    "https://helsinki-openapi.nuuka.cloud/api/v1/EnergyData/Daily/ListByProperty"
)

params_energy = {
    "Record": "propertyCode",
    "SearchString": propertyCodes[0],
    "ReportingGroup": "Electricity",
    "StartTime": "2020-01-01",
    "EndTime": TODAY,
}
response_energy = requests.get(endpoint_energy, params=params_energy)
print(response_energy.status_code)

df_all = pd.DataFrame.from_dict(response_energy.json())
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
# df_all = df_all.set_index(['timestamp'])
column_name = f"Building_{0}"
df_all = df_all.rename(columns={"value": column_name})
df_all = df_all.drop(["reportingGroup", "locationName", "unit"], axis=1)

names_positive_response = []

for i in range(len(propertyCodes)):
    endpoint_energy = (
        "https://helsinki-openapi.nuuka.cloud/api/v1/EnergyData/Daily/ListByProperty"
    )
    params_energy = {
        "Record": "propertyCode",
        "SearchString": propertyCodes[i],
        "ReportingGroup": "Electricity",
        "StartTime": "2020-01-01",
        "EndTime": TODAY,
    }
    response_energy = requests.get(endpoint_energy, params=params_energy)
    # print(response_energy.status_code)
    if response_energy.status_code != 200:
        pass
    else:
        if not response_energy.json():
            pass
        else:
            temp_df = pd.DataFrame.from_dict(response_energy.json())
            temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"])
            # temp_df = temp_df.set_index(['timestamp'])
            column_name = f"Building_{i}"
            temp_df = temp_df.rename(columns={"value": column_name})
            name = temp_df["locationName"][0]
            temp_df = temp_df.drop(["reportingGroup", "locationName", "unit"], axis=1)
            if temp_df["timestamp"].duplicated().sum() > 0:
                pass
            else:
                names_positive_response.append([i, name])
                df_all = df_all.merge(temp_df, how="outer", on=["timestamp"])
                print(f"--{i}", df_all.iloc[0, -1])


# %% [markdown] id="rd1NWBAEotju"
# Em um segundo momento, foi possível perceber que alguns edifícios tinham pouca quantidade de dados ou que não estavam mais sendo atualizados. Para isso foi aplicado uma segunda camada de filtro, para deixar apenas os edifícios que possuem dados relevantes.
# Em seguida foi feita a remoção de alguns valores que estavam muito discrepantes e substituídos usando o método `foward fill` do Pandas. Também foram eliminados os zeros absolutos (substituídos por 0.1) para evitar problemas nas métricas de predição.
#
# ///
#
# In a second moment, it was possible to notice that some buildings had little amount of data or that they were no longer being updated. For this, a second layer of filter was applied, to leave only the buildings that have relevant data.
# Next, some values that were clear outliers were removed and replaced using Pandas' `forward fill` method. Absolute zeros (replaced by 0.1) were also eliminated to avoid problems with prediction metrics.

# %% colab={"base_uri": "https://localhost:8080/"} id="R71p7EKsoi4j" outputId="2946cbdc-13d1-4fd1-8d5f-b212e049948d"
df_all_buildings = df_all.copy()
df_all_buildings = df_all_buildings.set_index(["timestamp"])
df_all_buildings.info()


# %% id="D7a9wAXF-mjN" colab={"base_uri": "https://localhost:8080/"} outputId="ad3c58e6-7222-488b-e524-f9e7aa8ae179"
for col in df_all_buildings.columns:
    if (df_all_buildings[col].isna().sum() > 500) or (
        df_all_buildings[col].tail(15).isna().sum() > 5
    ):
        df_all_buildings = df_all_buildings.drop(columns=[col])

df_all_buildings.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="5Rmfyw-YlxSX" outputId="36ceb009-9645-4ede-f1c2-02133eb46a6d"
df_all_buildings.plot(figsize=(22, 30), subplots=True)

# %% id="lKJjpAD3tR5g"
for col in df_all_buildings.columns:
    condition = col + " > 15000"
    if df_all_buildings.query(condition).shape[0] > 0:
        df_all_buildings[col] = df_all_buildings[col].apply(
            lambda x: np.nan if x > 15000 else x
        )
        df_all_buildings[col].fillna(method="ffill")
    df_all_buildings[col] = df_all_buildings[col].apply(lambda x: 10 if x < 10 else x)


# df_all_buildings.plot(figsize=(22, 30), subplots=True);

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="0SRS0YOxWzqx" outputId="6016120f-2d26-4add-c024-dda54f74f49e"
df_all_buildings["Building_3"].plot(figsize=(22, 20))

# %% [markdown] id="2wN9TlmZs6u4"
# ## Dados secundários - Climáticos
#
# Como dados auxiliares no processo de treinamento e previsão do consumo energético dos prédios, serão utilizados dados climáticos, que sabidamente possuem forte influência no consumo energético das edificações. Esses dados influenciam na utilização intensidade da utilização de sistemas de aquecimento e resfriamento do prédio, impactando seu consumo energético.
# Além disso, foram criadas outras duas colunas de dados, uma relativa ao mês do ano e outra à finais de semana. Em geral, na visualização dos gráficos, podemos observar vales constantes na forma das linhas, que indicam uma redução de consumo nos finais de semana, o que é previsível, dado à menor utilização dos edifícios. Os meses do ano também são importantes, pois estão ligados às estações do anos e as mudanças climáticas de cada período.
# Vale ressaltar que combinamos dados históricos com dados de previsão do tempo, que serão utilizados tanto no treinamento como na previsão do modelo.
#
# ///
# ## Secondary data - Weather
#
# As auxiliary data in the process of training and forecasting the energy consumption of buildings, climatic data will be used, which are known to have a strong influence on the energy consumption of buildings. These data influence the intensity of use of heating and cooling systems in the building, impacting its energy consumption.
# In addition, two other columns of data were created, one related to the month of the year and the other to the weekend. In general, when viewing the graphs, we can observe constant valleys in the form of lines, which indicate a reduction in consumption on weekends, which is predictable, given the lower use of buildings. The months of the year are also important as they are linked to the seasons and the climatic changes of each period.
# It is worth mentioning that we combine historical data with weather forecast data, which will be used both in training and forecasting the model.

# %% id="VjRuXwEcxcKd"
params = {
    "latitude": "60.19",
    "longitude": "24.94",
    "start_date": "2020-01-01",
    "end_date": TODAY,
    "hourly": ["temperature_2m", "relativehumidity_2m"],
    "timezone": "auto",
}
endpoint = "https://archive-api.open-meteo.com/v1/era5"
response = requests.get(endpoint, params=params)
response.json()["hourly"].keys()

response_dict = response.json()["hourly"]
df_hist_weather = pd.DataFrame.from_dict(response_dict)
df_hist_weather["time"] = pd.to_datetime(df_hist_weather["time"])
df_hist_weather.rename(columns={"time": "timestamp"}, inplace=True)
df_hist_weather.set_index(["timestamp"], inplace=True)
df_hist_weather = df_hist_weather.resample("D").mean()


# %% colab={"base_uri": "https://localhost:8080/", "height": 238} id="kA8Mkj6bC7fc" outputId="1c3a57b3-b80e-478a-da11-255daeb8be72"
df_hist_weather.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} id="kKh4U0lvEgrD" outputId="fefcdce2-5756-4d7d-b33e-0895834d83c7"
df_hist_weather = df_hist_weather[df_hist_weather["temperature_2m"].notna()]
df_hist_weather.tail(10)

# %% id="S6gWQ2ECDKmP"
params = {
    "latitude": "60.19",
    "longitude": "24.94",
    "hourly": ["temperature_2m", "relativehumidity_2m"],
    "past_days": 15,
    "timezone": "auto",
}
endpoint = "https://api.open-meteo.com/v1/forecast"
response = requests.get(endpoint, params=params)
response.json()["hourly"].keys()


response_dict = response.json()["hourly"]
df_fore_weather = pd.DataFrame.from_dict(response_dict)
df_fore_weather["time"] = pd.to_datetime(df_fore_weather["time"])
df_fore_weather.rename(columns={"time": "timestamp"}, inplace=True)
df_fore_weather.set_index(["timestamp"], inplace=True)
df_fore_weather = df_fore_weather.resample("D").mean()


# %% colab={"base_uri": "https://localhost:8080/", "height": 771} id="6DUXgR1_DcAW" outputId="badf04b5-d585-4d1f-dbf9-5cc951d8c85c"
df_fore_weather

# %% colab={"base_uri": "https://localhost:8080/"} id="KSBPurt4DvML" outputId="cc8e5886-0bc9-4720-9025-d02726526ebc"
temp_hist = df_hist_weather["temperature_2m"]
temp_fore = df_fore_weather["temperature_2m"]
temp_all = temp_hist.combine_first(temp_fore)
temp_all.tail(25)


# %% colab={"base_uri": "https://localhost:8080/"} id="obM6kng5D_p3" outputId="891506ec-751f-406e-f721-d46dd3b7662f"
humidity_hist = df_hist_weather["relativehumidity_2m"]
humidity_fore = df_fore_weather["relativehumidity_2m"]
humidity_all = humidity_hist.combine_first(humidity_fore)
humidity_all.tail(25)

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="gBGsNvl4K5CR" outputId="035c0123-7846-4a6b-fc59-1b3ef5c30064"
df_all_weather = pd.concat([temp_all, humidity_all], axis=1)
df_all_weather

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="y_8NZMoXNGt4" outputId="b9750ec6-f886-447f-d2b6-5f0d8327dde5"
df_all_weather["weekend"] = (df_all_weather.index.dayofweek >= 5).astype(int)
df_all_weather["month"] = df_all_weather.index.month

df_all_weather


# %% colab={"base_uri": "https://localhost:8080/", "height": 386} id="0zN4s5AXP04r" outputId="e97c75df-2804-45de-b910-32e5f255024d"
df_all_weather[["weekend", "month"]].plot(figsize=(22, 6))

# %% [markdown] id="NT6Mx3dH39tt"
# ## Trabalhando os dados com `TimeSeries` da biblioteca Darts
#
# A seguir os dados são processados a partir da transformação em classes específicas da biblioteca Darts, que são as `TimeSeries`. São criados dois grupos:
#   - Variáveis principais: a partir do Dataframe das Edificações, que são os dados que pretendemos fazer previsões
#   - Covariáveis: são os dados que influenciam nas variáveis principais, mas que não temos o objetivo de prever.
#   Nesse caso, utilizaremos as covariáveis como covariáveis futuras (na definição do Darst: `future covariates`). Isso porque esses dados possuem registros na mesma linha do tempo dos nossos dados principais, mas ainda acrescenta dados futuros, pois coletamos dados da previsão do tempo.
#
# Em seguida os dados são normalizados para o intervalo entre 0.0 e 1.0, que é mais adequado para o treinamento do modelo.
#
# ///
#
# ## Working with `TimeSeries` from Darts library
#
# Next, the data is processed by transforming it into specific classes of the Darts library, which are the `TimeSeries`. Two groups are created:
# - Main variables: from the Buildings Dataframe, which are the data that we intend to make predictions on.
# - Covariates: are the data that influence the main variables, but that we do not aim to predict.
# In this case, we will use the covariates as future covariates (in the Darts definition: `future covariates`). That's because this data has records on the same timeline as our main data, but it still adds future data as we collect weather forecast data.
#
# Then the data is normalized to the range between 0.0 and 1.0, which is more suitable for training the model.
#

# %% id="75FQgpdUc69Y"
series_buildings = TimeSeries.from_dataframe(df_all_buildings)
series_covariates = TimeSeries.from_dataframe(df_all_weather)

# %% colab={"base_uri": "https://localhost:8080/", "height": 539} id="ny6lwiPZdVai" outputId="a7a66ddb-d486-4777-cd0b-ae786ac337f5"
# O plot do Darts se limita a apenas 10 componentes
plt.figure(figsize=(16, 6), dpi=100)
series_buildings.plot()

# %% colab={"base_uri": "https://localhost:8080/", "height": 543} id="o9GmYmbWjGeE" outputId="23f7c741-8836-48ae-9119-868559783d5b"
scaler_buildings = Scaler(scaler=MinMaxScaler(feature_range=(0.005, 1)))
series_buildings_scaled = scaler_buildings.fit_transform(series_buildings)

plt.figure(figsize=(16, 6), dpi=100)
series_buildings_scaled.plot()

# %% colab={"base_uri": "https://localhost:8080/"} id="jZHmv8d5YTYB" outputId="ecbf0cf8-f2f5-4f17-ee7f-78a5b3b55575"
series = series_buildings_scaled["Building_3"].pd_series()
series[series <= 0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 524} id="gTlvwY1udcxU" outputId="f12a1fe6-816c-410f-d4af-b4a6ee81b379"
plt.figure(figsize=(16, 6), dpi=100)
series_covariates.plot()

# %% colab={"base_uri": "https://localhost:8080/", "height": 526} id="WN7Hpkc3jvZd" outputId="0084c908-1fcd-4c93-fffc-a1fc2435da1d"
scaler_covariates = Scaler()
series_covariates_scaled = scaler_covariates.fit_transform(series_covariates)

plt.figure(figsize=(16, 6), dpi=100)
series_covariates_scaled.plot()

# %% [markdown] id="LtL7ZZxx5zgD"
# Como input para o modelo, passaremos uma lista de séries (cada uma representando uma edificação) e as covariáveis precisam ser passadas também em uma lista no mesmo formato da lista dos dados principais.
# O modelo é criado com `lag=90` e `output_chunk_length=30`, ou seja, os dados são divididos em series de 90 entradas para 30 saídas, e vai sucessivamente avançando em ciclos até ler o conjunto completo.

# %% id="sOYvPqkVepVY"
model_test = LightGBMModel(
    lags=90,
    lags_future_covariates=(90, 1),
    output_chunk_length=30,
)

# %% id="RmrCzLXm3FgP" colab={"base_uri": "https://localhost:8080/"} outputId="76f79a09-b487-43ed-8cbe-dd3eb46ba072"
series_list_train = []
for col in series_buildings_scaled.columns:
    series_list_train.append(series_buildings_scaled[col][:-30])

len(series_list_train)


# %% colab={"base_uri": "https://localhost:8080/"} id="X10SSP9M6_Y2" outputId="6708c68c-6c9a-4796-ca17-6d9730bc55a8"
series_list_test = []
for col in series_buildings_scaled.columns:
    series_list_test.append(series_buildings_scaled[col][-30:])


len(series_list_test)


# %% colab={"base_uri": "https://localhost:8080/"} id="mohak-xORWU1" outputId="76bcf17e-ae3d-4ad6-812d-f0b6bcf4003d"
future = series_covariates_scaled

covariates_list = []
for i in range(len(series_list_train)):
    covariates_list.append(future)
len(covariates_list)

# %% colab={"base_uri": "https://localhost:8080/"} id="Wtcg5o6ueqjM" outputId="a4f4be6b-c84b-4035-cee7-07bf18ba073b"
model_test.fit(
    series=series_list_train,
    future_covariates=covariates_list,
)

# %% [markdown] id="VYDr_rMaIZcm"
# ### Resultados
#
# Após o treinamento do modelo, foi criado o gráfico abaixo com a comparação entre a previsão e os testes. É possível perceber que o modelo foi relativamente bem em grande parte dos casos, mas em alguns ficou mais destoante. É possível que isso se deva ao fato de que algumas dessas séries tem formatos mais aleatórios. Uma possibilidade a ser testada é treinar o modelo com uma série menor - de 1 ano, ao invés de mais de 2 anos.
#
# Em seguida, foi utilizado a métrica de RMSE para cada um dos edifícios.
#
# ///
# ### Results
# After training the model, the graph below was created with the comparison between the prediction and the tests. It is possible to see that the model performed relatively well in most cases, but in some cases it became more discordant. It is possible that this is due to the fact that some of these series have more random shapes. One possibility to test is to train the model with a shorter series - 1 year, instead of more than 2 years.
#
# Next, the RMSE metric was used for each of the buildings.
#


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3_D8tOJsfbWi" outputId="18f94321-688c-4f61-c868-8f0924777a7c"
figure = plt.figure(1, figsize=(20, 56))

for i in range(len(series_list_train)):
    ax = plt.subplot(16, 2, i + 1)
    series = series_list_train[i]
    test = series_list_test[i]
    pred = model_test.predict(30, series=series, future_covariates=future)
    series[-90:].plot(label="train")
    test.plot(label="test")
    pred.plot(label="predicted")
    plt.title(series.columns[0])
    ax.legend(loc="upper left", frameon=True, framealpha=1.0)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1.8)


# %% colab={"base_uri": "https://localhost:8080/"} id="YG-KBLDiAT6O" outputId="cd27fa1f-3b9b-40dc-8761-b15c7efbe901"
for i in range(len(series_list_train)):
    series = series_list_train[i]
    pred = model_test.predict(30, series=series, future_covariates=future)
    test_result = rmse(series_list_test[i], pred)
    print(
        f"The RMSE for Buildin {series_list_train[i].columns[0]} is: {test_result:.2f}"
    )


# %% [markdown] id="yejQGs8MAS3Z"
# ---

# %% [markdown] id="m4rk5wWbkk2_"
# ## Treinando o modelo com os dados completos
#
# O modelo agora será treinado com os dados completos, sem a divisão entre treino e teste. Vale lembrar que, mesmo utilizando os dados completos, as covariantes futuras contêm registros de previsão do tempo que estão à frente da linha do tempo de consumo energético, o que será ajudará no desempenho das previsões.
#
# Após o treinamento do modelo, é feito o testo com um edifício aleatório para testar a previsão.
#
# ///
# ## Training the model with the whole data
# The model will now be trained on the complete data, without the split between training and testing. It is worth remembering that, even using the complete data, the future covariants contain weather forecast records that are ahead of the energy consumption timeline, which will help in the performance of the forecasts.
#
# After training the model, it is tested with a random building to test the prediction.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="dLBJ2fzIk1Hg" outputId="e1ad1922-17d7-41c4-b65c-3fc41f3b589e"
series_all_train = []
for col in series_buildings_scaled.columns:
    series_all_train.append(series_buildings_scaled[col])

len(series_all_train)

# %% id="hZo62dX-smEH"
model = LightGBMModel(
    lags=90,
    lags_future_covariates=(90, 1),
    output_chunk_length=30,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="hnwaEbbplCCq" outputId="5fd170d0-42f5-4784-e9b5-0afdc9439fc8"
model.fit(
    series=series_all_train,
    future_covariates=covariates_list,
)

# %% id="cFxAFl_b4VuU" colab={"base_uri": "https://localhost:8080/", "height": 472} outputId="ce073c23-cf8e-4a75-c836-09641b9925f9"
endpoint_energy = (
    "https://helsinki-openapi.nuuka.cloud/api/v1/EnergyData/Daily/ListByProperty"
)

params_energy = {
    "Record": "propertyCode",
    "SearchString": propertyCodes[5],
    "ReportingGroup": "Electricity",
    "StartTime": "2020-01-01",
    "EndTime": TODAY,
}
response_energy = requests.get(endpoint_energy, params=params_energy)
print(response_energy.status_code)

new_building = pd.DataFrame.from_dict(response_energy.json())
new_building["timestamp"] = pd.to_datetime(new_building["timestamp"])
new_building = new_building.set_index(["timestamp"])
column_name = f"Building {0}"
new_building = new_building.rename(columns={"value": column_name})
new_building = new_building.drop(["reportingGroup", "locationName", "unit"], axis=1)


new_building


# %% colab={"base_uri": "https://localhost:8080/", "height": 340} id="p8sJ0ig0kLtR" outputId="40b23e7e-c140-4400-fe7b-c0465e32dd95"
series = TimeSeries.from_dataframe(new_building, fill_missing_dates=True)
series_scaled = scaler_buildings.fit_transform(series)
pred = model.predict(10, series=series_scaled, future_covariates=future)
# series_scaled[-90:].plot()
# pred.plot()

scaler_inverse = Scaler()
series_inverted = scaler_inverse.fit(series)
series_inverted = scaler_inverse.inverse_transform(series_scaled)
series_inverted[-90:].plot(label="train")
pred_inverted = scaler_inverse.inverse_transform(pred)
pred_inverted.plot(label="prediction")
print(model)


# %% id="CRfIF15733AU"
with bz2.BZ2File("models/ts_model" + ".pbz2", "w") as f:
    pickle.dump(model, f)

# model.save('ts_model.pkl')

# %% id="5R7QvQF94dyi"
future.to_pickle("models/future_covariates.pkl")

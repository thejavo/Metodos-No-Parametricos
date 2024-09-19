import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
#import random
import pandas as pd
import os
import requests
from io import BytesIO
#import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import warnings
#from colorama import Fore, Style

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def log(mensaje: str, nuevalinea = True):

    texto = f"{datetime.now().strftime('%H:%M:%S')} - {mensaje}"
    if nuevalinea:
        print(texto)
    else:
        sys.stdout.write("\r" + texto)
        sys.stdout.flush()

def gettickerinfo(tickers: list, archivo="tickers.csv") -> pd.DataFrame:
    tickers_df = pd.DataFrame()

    if os.path.exists(archivo):
        tickers_df = pd.read_csv(archivo)
        tickers = list(set(tickers) - set(tickers_df['Ticker'].tolist()))

    for ticker in tickers:
        tickers_df = tickers_df._append({'Ticker': ticker, 'longName': yf.Ticker(ticker).info['longName']},ignore_index=True)

    tickers_df.to_csv(archivo, index=False)

    return tickers_df


def getTickers(archivo = 'tickers.csv',group = 'GSPC') -> list:
    respuesta: list = []
    if os.path.exists(archivo):
        respuesta = pd.read_csv(archivo)['Ticker'].tolist()
    else:
        if group == 'GSPC':
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
            tickers.to_csv(archivo, index=False)
            respuesta = tickers['Symbol'].ToList()

        elif (group == 'RUT'):
            url = "https://www.beatthemarketanalyzer.com/blog/wp-content/uploads/Russell-2000-Stock-Tickers-List.xlsx"

            # Necesito definir encabezados http para que el sitio crea que somos un browser y nos deje descargar el archivo
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36'}

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                excelcontent = BytesIO(response.content)
                df = pd.read_excel(excelcontent)
                tickers = (df.drop(index=[0, 1, 2]).drop(columns=[df.columns[1], df.columns[2]]))
                tickers.rename(columns={'Unnamed: 0': 'Tickers'}, inplace=True)

                tickers.to_csv(archivo, index=False)
                respuesta = tickers['Tickers'].tolist()

    return respuesta


def guardainfo(df: pd.DataFrame, archivoname, ind = False):
    try:
        with open(archivoname+'.csv', 'x') as f:
            df.to_csv(f, header=True, index=ind)
    except FileExistsError:
        with open(archivoname+'.csv', 'a') as f:
            df.to_csv(f, header=False, index=ind)


#--- PARAMETROS DE LA APLICACION ---
fecha_dde = datetime(2019, 1, 1)
fecha_hta = datetime(2024, 1, 1)
#--- FIN PARAMETROS DE LA APLICACION ---

tickers = []

date_range = pd.date_range(start=fecha_dde, end=fecha_hta)
precios = pd.DataFrame()
precios_tmp = pd.DataFrame()

tickers = ['^GSPC']
tickers.extend(getTickers(archivo='tickerstesis.csv'))

precios_tmp = yf.download(tickers=tickers, start=fecha_dde, end=fecha_hta, progress=False)['Adj Close']
if isinstance(precios_tmp, pd.Series):
    precios_tmp = precios_tmp.to_frame().rename(columns={'Adj Close': tickers[0]})


if precios.empty:
    precios = precios_tmp
elif precios_tmp.index.equals(precios.index):
    precios = precios.join(precios_tmp[precios_tmp.columns.difference(precios.columns)], sort=True)

# guardamos los precios obtenidos en un .csv y los tickers con problemas en otro.
precios.to_csv('precios.csv', index=True)


retornos = (np.log(precios / precios.shift(1))).iloc[1:]

# marco el dataframe de precios, que es muy grande y ya no es necesario, para liberar memoria
del precios
del precios_tmp

tickers = retornos.columns.tolist()
tickerswospy = [t for t in tickers if t != '^GSPC']
tickers_df = gettickerinfo(tickers=tickers)
dias = len(retornos)
fecha_dde = retornos.index[0]
fecha_hta = retornos.index[-1]

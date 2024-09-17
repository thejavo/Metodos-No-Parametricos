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

def plusTickers(tickers_actuales, tickers_adicionales, total_necesario) -> list:

    nuevos_tickers = set(tickers_adicionales) - set(tickers_actuales)
    plustickers = list(nuevos_tickers)[:(total_necesario - len(tickers_actuales))]

    return plustickers

def vectorgamma(largo, unos):
    gamma = []
    if unos > largo:
        raise ValueError("no puede haber mas unos que el largo del vector")

    gamma = [1] * unos + [0] * (largo - unos)
    return np.array(gamma).reshape(largo,1)

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
ventana_estimacion = 250
ventana_evento = 10
activos_totales = 25
activos_muestra = 5
escenarios = 1000
q_de_gammas = 4
k = 0.5   # 0,5 - 1 - 2
Lambda = 0.1 # 0,1 - 1 - 10
#--- FIN PARAMETROS DE LA APLICACION ---

nivel_confianza = 0.05
# Datos de la distribucion t de student para el test de hipotesis parametrica
grados_libertad = ventana_estimacion - 2
nivel_confianza_2 = nivel_confianza / 2
valor_critico_t = stats.t.ppf(nivel_confianza, grados_libertad)
valor_critico_t2 = stats.t.ppf(nivel_confianza_2, grados_libertad)

# Datos de la distribucion normal standard
valor_critico_n = stats.norm.ppf(nivel_confianza)
valor_critico_n2 = stats.norm.ppf(nivel_confianza_2)

tickers = []
#tickersRUT = getTickersRUT()
#wrongtickers = []
intentos = 1

#date_range = pd.date_range(start=fecha_dde, end=fecha_hta)
precios = pd.DataFrame()
precios_tmp = pd.DataFrame()

""""
#En caso de que ya hubiesemos guardamos un archivo de precios
#lo levantamos a un dataframe y cotejamos que tengamos la cantidad de activos
#que requiere el analisis
if os.path.exists('precios.csv'):
    precios = pd.read_csv('precios.csv', index_col='Date', parse_dates=['Date'])
    tickersRUT = list(set(tickersRUT) - set(precios.columns.tolist()))

    #si entramos por este camino, es probable que ya tengamos un listado de
    #tickers que no cumplieron con nuestras especificaciones, lo levantamos
    #y los quitamos del listado de tickets del Russell2000
    if (os.path.exists('wrongtickers.csv')):
        wrongtickers = (pd.read_csv('wrongtickers.csv')['Tickers']).tolist()
        tickersRUT = list(set(tickersRUT) - set(wrongtickers))

    test = True if precios.shape[1] < activos_totales else False
else:
    #en caso de arrancar de 0 tomamos la lista del S&P500, le sumamos la lista del Russell2000
    #y la recortamos al tamaño que necesitamos.
    test = True
    tickers = ['^GSPC']
    tickers.extend(getTickersGSPC())
    tickers = (tickers + tickersRUT)[:activos_totales]
"""

if os.path.exists('precios.csv'):
    precios = pd.read_csv('precios.csv', index_col='Date', parse_dates=['Date'])
else:
    tickers = ['^GSPC']
    tickers.extend(getTickers(archivo='tickerstesis.csv'))
    test = True


#La variable test define si tenemos que ir a buscar activos a YahooFinance.
#la variable intentos evitará que nos quedemos encerrados en el bucle en caso
#de no poder completar la cantidad de activos requeridos.
while test and intentos > 0:
    test = False
    intentos -= 1

    #determinamos que tickers vamos a bajar de YahooFinance, seran los que estan en la lista de tickers
    #y todavia no tenemos en los precios bajados.
    #if precios.shape[1] > 0:
    #    dntickers = plusTickers(precios.columns.tolist(),tickersRUT,1000)
    #else:
    dntickers = tickers

    precios_tmp = yf.download(tickers=dntickers, start=fecha_dde, end=fecha_hta, progress=False)['Adj Close']
    if isinstance(precios_tmp, pd.Series):
        precios_tmp = precios_tmp.to_frame().rename(columns={'Adj Close': dntickers[0]})

    #guardamos en una lista aquellos tickers que tuvieron al menos un valor nulo para el periodo,
    #y los borramos de la lista de tickers y de la lista de los tickers de Russell2000
#    wrongtickers_set = set(wrongtickers)
#    if precios_tmp.empty or not precios_tmp.index.equals(precios.index):
#        wrongtickers_set.update(precios_tmp.columns.tolist())
#    else:
#        wrongtickers_set.update(precios_tmp.columns[precios_tmp.isnull().any()].tolist())
#    wrongtickers = list(wrongtickers_set)

#    if len(wrongtickers)>0:
#        tickersRUT = list(set(tickersRUT) - set(wrongtickers))

    #borramos los tickers con valores nulos y en caso de que el rango devuelto por
    # yfinance contenga todas las fechas lo sumamos lo que queda al dataframe de precios.
    precios_tmp = precios_tmp.dropna(axis=1, how='any')
    if precios.empty:
        precios = precios_tmp
    else:
        if precios_tmp.index.equals(precios.index):
            precios = precios.join(precios_tmp[precios_tmp.columns.difference(precios.columns)], sort=True)
            # guardamos los precios obtenidos en un .csv y los tickers con problemas en otro.
            precios.to_csv('precios.csv', index=True)

    pd.DataFrame(wrongtickers, columns=['Tickers']).to_csv('wrongtickers.csv', index=False)

    #si no alcanzamos la cantidad de activos necesaria, seguimos en carrera
    if (precios.shape[1] < activos_totales):
        test = True

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

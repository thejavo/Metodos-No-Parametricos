import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
import numpy as np
import random

def vectorgamma(largo, unos):
    gamma = []
    if unos > largo:
        raise ValueError("no puede haber mas unos que el largo del vector")

    gamma = [1] * unos + [0] * (largo - unos)
    return np.array(gamma).reshape(largo,1)

#--- PARAMETROS DE LA APLICACION ---

fecha_evento=datetime(2020, 6, 8)
ventana_estimacion = 101
gap_estimacion_evento = 10
pre_evento = 3
ventana_evento = 10

fecha_finestimacion = fecha_evento - gap_estimacion_evento


q_de_gammas = 4
k = 0.5       # 0,5 - 1 -  2
Lambda = 0.1  # 0,1 - 1 - 10
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


if os.path.exists('precios.csv'):
    precios = pd.read_csv('precios.csv', index_col='Date', parse_dates=['Date'])
else:
    sys.exit("Falta el archivo precios.csv")

retornos = (np.log(precios / precios.shift(1))).iloc[1:]
tickers = retornos.columns.tolist()

del precios

gamma = vectorgamma(ventana_evento, q_de_gammas)
scar_aux, J1, J2, ZR, GS = [], [], [], [], []


fecha_finestimacion = retornos.iloc[fecha_evento - 1].name
fecha_estimacion = retornos.iloc[fecha_evento - gap_estimacion_evento - ventana_estimacion].name
fecha_finestudio = retornos.iloc[fecha_evento + (ventana_evento - 1)].name

ticker_rnd = random.choice(tickers)

# Datos de retornos del evento.
est_window = retornos.loc[fecha_estimacion:fecha_finestimacion, ['^GSPC', ticker_rnd]].rename(
    columns={ticker_rnd: 'ticker_rnd'}).reset_index()
evt_window = retornos.loc[fecha_evento:fecha_finestudio, ['^GSPC', ticker_rnd]].rename(
    columns={ticker_rnd: 'ticker_rnd'}).reset_index()

# Matriz X con una columna de unos (para el término constante alfa) y los retornos del mercado
X = np.column_stack((np.ones(len(est_window)), est_window['^GSPC'].values))

# Vector Y con los retornos del activo
Y = np.array(est_window['ticker_rnd'].values).reshape(-1, 1)  # <--- lo transformamos en un vector columna

# MARKET MODEL - estimation window (Ri = alpha + Beta.Rm)

# Calcular los parámetros alfa y beta usando la fórmula de MCO: (X'X)^(-1)X'Y
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

est_window['Ticker'] = ticker_rnd
est_window['L'] = 1
est_window['Rm'] = theta_hat[0][0] + est_window['^GSPC'] * theta_hat[1][0]
est_window['e_hat'] = est_window['ticker_rnd'] - est_window['Rm']
e = (est_window['e_hat'].values).reshape(-1, 1)
DVO_e_hat = (1 / (len(est_window) - 2) * (e.T @ e))[0][0]
# ESP_e_hat = np.mean(est_window['e_hat'])
VAR_theta_hat = np.linalg.inv(X.T @ X) * DVO_e_hat

# Event Window
evt_window['Ticker'] = ticker_rnd
evt_window['L'] = 2
evt_window['Rm'] = theta_hat[0][0] + evt_window['^GSPC'] * theta_hat[1][0]  # usamos el theta de la estimation window
evt_window['e_hat'] = evt_window['ticker_rnd'] - evt_window['Rm']

S0 = k * DVO_e_hat  # --------------->  PARA SHOCK
evt_window['e_hat'] += S0 * np.exp(-Lambda * evt_window.index)

#FUNCION DE SHOCK tiene la forma S0 * e^-lambda * t . lambda es la constante de atenuacion.
# donde S0 es funcion del desvio std multiplicado por un factor k
# S0 = k * sigma, k toma por lo general valores (0.1, 0.5, 1, 2)
# que pasa si lambda es 0? entonces S0 se transforma en un shock permanente k * sigma.
# De esto se concluye que cuanto mas alto sea k y mas bajo sea lambda, mas facilmente
# se podra detectar el efecto del evento.
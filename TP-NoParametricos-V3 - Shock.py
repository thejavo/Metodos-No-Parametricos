# -*- coding: utf-8 -*-
"""
TP Metodos No Parametricos

Ramiro Nievas
A.Javier Sanchez
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import random
import pandas as pd
import os
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import warnings
from colorama import Fore, Style

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

def getTickersGSPC(archivo = 'tickersgspc.csv') -> list:
    if os.path.exists(archivo):
        tickers = pd.read_csv(archivo)
    else:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tickers = []
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        tickers.to_csv(archivo, index=False)

    return tickers['Symbol'].tolist()

def getTickersRUT(archivo = 'tickersrut.csv') -> list:
    if os.path.exists(archivo):
        tickers = pd.read_csv(archivo)
    else:
        url = "https://www.beatthemarketanalyzer.com/blog/wp-content/uploads/Russell-2000-Stock-Tickers-List.xlsx"

        # Necesito definir encabezados http para que el sitio crea que somos un browser y nos deje descargar el archivo
        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36'}
        tickers = []

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            excelcontent = BytesIO(response.content)
            df = pd.read_excel(excelcontent)
            tickers = (df.drop(index=[0, 1, 2]).drop(columns=[df.columns[1], df.columns[2]]))
            tickers.rename(columns={'Unnamed: 0': 'Tickers'}, inplace=True)

            tickers.to_csv(archivo, index=False)

    return tickers['Tickers'].tolist()


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
fecha_dde = datetime(2010, 1, 1)
fecha_hta = datetime(2017, 1, 1)
ventana_estimacion = 250
ventana_evento = 10
activos_totales = 1000
activos_muestra = 100
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
tickersRUT = getTickersRUT()
wrongtickers = []
intentos = 3

#date_range = pd.date_range(start=fecha_dde, end=fecha_hta)
precios = pd.DataFrame()
precios_tmp = pd.DataFrame()

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

#La variable test define si tenemos que ir a buscar activos a YahooFinance.
#la variable intentos evitará que nos quedemos encerrados en el bucle en caso
#de no poder completar la cantidad de activos requeridos.
while test and intentos > 0:
    test = False
    intentos -= 1

    #determinamos que tickers vamos a bajar de YahooFinance, seran los que estan en la lista de tickers
    #y todavia no tenemos en los precios bajados.
    if precios.shape[1] > 0:
        dntickers = plusTickers(precios.columns.tolist(),tickersRUT,1000)
    else:
        dntickers = tickers

    precios_tmp = yf.download(tickers=dntickers, start=fecha_dde, end=fecha_hta, progress=False)['Adj Close']
    if isinstance(precios_tmp, pd.Series):
        precios_tmp = precios_tmp.to_frame().rename(columns={'Adj Close': dntickers[0]})

    #guardamos en una lista aquellos tickers que tuvieron al menos un valor nulo para el periodo,
    #y los borramos de la lista de tickers y de la lista de los tickers de Russell2000
    wrongtickers_set = set(wrongtickers)
    if precios_tmp.empty or not precios_tmp.index.equals(precios.index):
        wrongtickers_set.update(precios_tmp.columns.tolist())
    else:
        wrongtickers_set.update(precios_tmp.columns[precios_tmp.isnull().any()].tolist())
    wrongtickers = list(wrongtickers_set)

    if len(wrongtickers)>0:
        tickersRUT = list(set(tickersRUT) - set(wrongtickers))

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

log("=====  Comienza ciclo de escenarios  =====")

gamma = vectorgamma(ventana_evento, q_de_gammas)

scar_aux, J1, J2, ZR, GS = [], [], [], [], []

for escenario in range(1, escenarios + 1):

    event_data = pd.DataFrame()
    estimationWindows = pd.DataFrame()
    eventWindows = pd.DataFrame()
    suma_var_e_hat_star = None
    fullWindow = pd.DataFrame()
    l2_rank = pd.DataFrame()

    for muestra in range(1, activos_muestra+1):

        #el dia random arrancara desde 251 hasta muestra total - 10, de manera de poder tener siempre
        #una ventana de estimacion y 10 dias posteriores.
        dia_rnd = random.randint(ventana_estimacion + 1, dias - ventana_evento)
        #utilizamos iloc, para ubicar la fecha segun el numero de fila correspondiente
        fecha_evento_rnd = retornos.iloc[dia_rnd].name

        fecha_finestimacion = retornos.iloc[dia_rnd-1].name
        fecha_estimacion = retornos.iloc[dia_rnd-ventana_estimacion].name
        fecha_finestudio = retornos.iloc[dia_rnd+(ventana_evento-1)].name

        ticker_rnd = random.choice(tickerswospy)

        # Datos de retornos del evento.
        est_window = retornos.loc[fecha_estimacion:fecha_finestimacion, ['^GSPC', ticker_rnd]].rename(columns={ticker_rnd: 'ticker_rnd'}).reset_index()
        evt_window = retornos.loc[fecha_evento_rnd:fecha_finestudio, ['^GSPC', ticker_rnd]].rename(columns={ticker_rnd: 'ticker_rnd'}).reset_index()

        # Matriz X con una columna de unos (para el término constante alfa) y los retornos del mercado
        X = np.column_stack((np.ones(len(est_window)), est_window['^GSPC'].values))

        # Vector Y con los retornos del activo
        Y = np.array(est_window['ticker_rnd'].values).reshape(-1,1) #<--- lo transformamos en un vector columna

        # MARKET MODEL - estimation window (Ri = alpha + Beta.Rm)

        # Calcular los parámetros alfa y beta usando la fórmula de MCO: (X'X)^(-1)X'Y
        theta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

        est_window['Escenario'] = escenario
        est_window['Evento'] = muestra
        est_window['Ticker'] = ticker_rnd
        est_window['L'] = 1
        est_window['Rm'] = theta_hat[0][0] + est_window['^GSPC'] * theta_hat[1][0]
        est_window['e_hat'] = est_window['ticker_rnd'] - est_window['Rm']
        e = (est_window['e_hat'].values).reshape(-1,1)
        DVO_e_hat = (1 / (len(est_window) - 2) * (e.T @ e))[0][0]
        #ESP_e_hat = np.mean(est_window['e_hat'])
        VAR_theta_hat = np.linalg.inv(X.T @ X) * DVO_e_hat

        # Event Window
        evt_window['Escenario'] = escenario
        evt_window['Evento'] = muestra
        evt_window['Ticker'] = ticker_rnd
        evt_window['L'] = 2
        evt_window['Rm'] = theta_hat[0][0] + evt_window['^GSPC'] * theta_hat[1][0]  # usamos el theta de la estimation window
        evt_window['e_hat'] = evt_window['ticker_rnd'] - evt_window['Rm']
      
        S0 = k * DVO_e_hat # --------------->  PARA SHOCK
        evt_window['e_hat'] += S0 * np.exp(-Lambda * evt_window.index)
        
        #ESP_e_hat_star = np.mean(evt_window['e_hat']) CONTROL

        X_asterix = np.column_stack((np.ones(len(evt_window)), evt_window['^GSPC'].values))

        VAR_e_hat_star = DVO_e_hat * (np.eye(len(evt_window)) + X_asterix @ np.linalg.inv(X.T @ X) @ X_asterix.T) #Matriz de Covarizanzas Vi
        # Esto es la matriz de covarianzas de los errores en la ventana del evento. es decir VAR_evt = E[e* @ e*.T]
        # recordar que (A @ B).T = B.T @ A.T
        # Voy sumando las matrices de Covarianzas de cada evento dentro del mismo escenario

        suma_var_e_hat_star = VAR_e_hat_star if suma_var_e_hat_star is None else suma_var_e_hat_star + VAR_e_hat_star

        # Ahora empiezan los test de hipotesis. Lo primero es introducir supuestos estadisticos.
        # el primer supuesto es que, bajo H0 los retornos anormales en la event window son "normales multivariados"
        # e*_evt ~ N(0,VAR_evt). Vamos a utilizar ahora los CAR, sumando los e* en el tiempo, tambien podrian acumularse
        # across events.

        CAR = (gamma.T @ evt_window['e_hat'])[0]
        VAR_CAR = (gamma.T @ VAR_e_hat_star @ gamma)[0][0]

        # Los CAR tambien los supondremos N(0,gamma.T @ Var(ê*) @ gamma) y entonces los estandarizamos
        SCAR = CAR / (VAR_CAR ** 0.5)
        scar_aux.append(SCAR)

        VAR_SCAR = (len(est_window) - 2) / (len(est_window) - 4)
        # Estos SCAR se distribuyen como una t de student con L1-2 grados de libertad.
        # Al comienzo del programa esta calculado el valor critico para la t de student

        #Esto es para los test de Rankings
        # Concatenar los DataFrames uno debajo del otro
        fullWindow = pd.concat([fullWindow, est_window,evt_window])

        # Calcular el rango para el evento específico
        rango = fullWindow[fullWindow["Evento"] == muestra]['e_hat'].rank()
        fullWindow.loc[fullWindow["Evento"] == muestra, 'rank'] = rango
        del rango

        event_data = event_data._append({"Escenario": escenario,
                                            "Evento": muestra,
                                            "Ticker": ticker_rnd,
                                            "Fecha Evt": fecha_evento_rnd,
                                            "alpha": round(theta_hat[0][0], 5),
                                            "beta": round(theta_hat[1][0], 5),
                                            "DVO_e_hat" : DVO_e_hat,
                                            "Var_e_hat_star": [matriz.tolist() for matriz in VAR_e_hat_star],
                                            "CAR": CAR,
                                            "Var_CAR": VAR_CAR,
                                            "SCAR": SCAR,
                                         }, ignore_index=True)

        log(f"Esc {escenario}/Evt {muestra} {fecha_evento_rnd.date()}, Est.W.: {fecha_estimacion.date()} - {ticker_rnd} ({tickers_df.loc[tickers_df['Ticker'] == ticker_rnd, 'longName'].values[0]}) ---> alfa = {round(theta_hat[0][0], 5)}, Beta = {round(theta_hat[1][0], 3)}, - SCAR: {SCAR}", False)


    # Ahora, estos SCAR obtenidos, diremos que son independientes e identicamente distribuidos (iid), Tendremos entonces
    # N SCAR, del tipo t l1-2 e iid. Podemos entonces obtener un SCAR promedio. OJO, ACLARA EN CLASE, TODO ESTO FUNCIONA
    # BAJO EL HECHO DE QUE ESTAMOS AGREGANDO SCAR CON EL MISMO GAMMA. (vuelo el calculo aleatorio de gamma)
    # Con esto vamos a estandarizar la distribucion y obtener J2 = N**0.5 * SCAR_PROM / (VAR_SCAR ** 0.5)

    #scar_aux = pd.concat([scar_aux,event_data["Event Window"].apply(lambda  fila: fila["SCAR"])])

    SCAR_PROM = event_data["SCAR"].mean()
    J2.append((activos_muestra ** 0.5) * SCAR_PROM / (VAR_SCAR ** 0.5))

    # Otra cosa que podemos hacer es promediar los abnormal returns across the events por dia. Es decir el promedio de
    # todos los abnormal returns para el dia 0, para el dia 1, etc. Estos retornos tendran volatilidades diferentes, acorde
    # a las volatilidades de los activos que representan, es por eso que esta medida, quizas pueda ser dominada por los
    # activos con mayor volatilidad, es decir es un promedio "tendencioso" (esto lo digo yo), maybe. Es decir, cuando promediamos
    # intentamos limpiar la señal comun a todos los eventos removiendo el ruido. para promediar estos retornos usaremos
    # prom_e* = 1/eventos * (sum i=1 -> eventos) ê*i  Agregamos luesgo estos promedios via gamma. igual que antes y obtenedremos
    # el prom_CAR = gamma.T @ prom_e*  Tambien podremos calcular la VAR_prom_CAR = gammma.T @ VAR_prom_e* @ gamma
    # la VAR_prom_e* = 1/eventos**2 * (sum i=1 -> eventos) VAR_e_hat_star
    # con todo esto vamos a sacar nuestro estadístico que sera el prom_CAR_std = prom_CAR / VAR_prom_CAR ** 0.5

    # Algunas cositas, los Dataframes hacen agua cuando se trata de estructuras anidadas, si nos queda tiempo vamos a implementar
    # una base SQLite para guardar adecuadamente la información y poder accederla sin tantos problemas
    # Extraemos los abnormal returns a nivel de los Escenarios para poder manipularlos.

    # Si bien es extraño, dado que las Varianzas de los PROM_e_hat_star dependen de sumar las Varianzas de los e_hat_star
    # en cada Escenario entonces podemos calcularla antes de tener los PROM_e_hat_star

    # Al final del bucle, suma_var_e_hat_star contendrá la suma de todas las matrices "Var(ê*)"
    VAR_PROM_e_hat_star = (1 / activos_muestra ** 2) * suma_var_e_hat_star

    # Agrupamos los abnormal returns por dia de evento, los del dia 0, el 1, etc, across events
    PROM_e_hat_star = fullWindow[fullWindow["L"] == 2].groupby([fullWindow[fullWindow["L"] == 2].index])["e_hat"].mean()

    PROM_CAR = (gamma.T @ PROM_e_hat_star)[0]
    J1.append((PROM_CAR / (gamma.T @ VAR_PROM_e_hat_star @ gamma)**0.5)[0][0])

    # Acumulo para test de ranking
    l2_rank = pd.concat([fullWindow[fullWindow['L'] == 2]], ignore_index=False)

    #log("Calculando K_hat promedio para cada dia across events")
    PROM_K_hat = pd.concat([fullWindow[fullWindow['L'] == 1].groupby([fullWindow[fullWindow['L'] == 1].index])["rank"].mean(),fullWindow[fullWindow['L'] == 2].groupby([fullWindow[fullWindow['L'] == 2].index])["rank"].mean()])
    #log("Calculando K_L2_hat promedio para cada dia across events")
    KL2 = l2_rank.groupby([l2_rank.index])["rank"].mean()
    #log("Promediando todos los K_L2_hat para cada Escenario")
    PROM_KL2 = KL2.mean()

    ACU_K_HAT = 0
    for k_hat_aux in PROM_K_hat:
        ACU_K_HAT += (k_hat_aux - (ventana_estimacion+ventana_evento+1)/2) ** 2

    DVO_PROM_K_hat = (1/(ventana_estimacion+ventana_evento+1) * ACU_K_HAT) ** 0.5

    ZR.append(ventana_evento ** 0.5 * (PROM_KL2 - (ventana_estimacion+ventana_evento+1)/2) / DVO_PROM_K_hat)

    #Test de signo
    fullWindow['signo'] = fullWindow.apply(lambda x: 1 if x['e_hat'] > 0 else 0, axis=1)
    PROM_DIA_P_hat = fullWindow[fullWindow['L'] == 1].groupby([fullWindow[fullWindow['L'] == 1].index])["signo"].mean()
    P_hat = PROM_DIA_P_hat.mean()
    X_evt = fullWindow[(fullWindow["L"] == 2) & (fullWindow.index == 0)]["signo"].sum()
    GS.append((X_evt - activos_muestra * P_hat) / (activos_muestra * P_hat * (1-P_hat)) ** 0.5)

    guardainfo(event_data, 'eventos')
    guardainfo(fullWindow, 'fullwindows', True)

guardainfo(pd.DataFrame({"J1": J1,
              "J2": J2,
              "ZR": ZR,
              "GS": GS }),"escenarios", True)
print()
log("=====  Tests paramétricos  =====")
log("")
log("**Cola Izquierda**")
#log(f"SCAR: El nivel de rechazo (debajo del {nivel_confianza * 100}%) para {escenarios * activos_muestra} eventos es de {np.sum(scar_aux < valor_critico_t) * 100 / (escenarios * activos_muestra)} %")
log(f"J2 : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(J2 > valor_critico_n) * 100 / escenarios }%")
log(f"J1 : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(J1 > valor_critico_n) * 100 / escenarios }%")
log("**Cola derecha**")
#log(f"SCAR: El nivel de rechazo (debajo del {nivel_confianza * 100}%) para {escenarios * activos_muestra} eventos es de {np.sum(scar_aux > -valor_critico_t) * 100 / (escenarios * activos_muestra)} %")
log(f"J2 : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(J2 < -valor_critico_n) * 100 / escenarios }%")
log(f"J1 : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(J1 < -valor_critico_n) * 100 / escenarios }%")
log("**A 2 colas**")
#log(f"SCAR: El nivel de rechazo (debajo del {nivel_confianza * 100}%) para {escenarios * activos_muestra} eventos es de {(np.sum(scar_aux < valor_critico_t2) + np.sum(scar_aux > -valor_critico_t2)) * 100 / (escenarios * activos_muestra)} %")
log(f"J2 : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {(np.sum(J2 > valor_critico_n2) + np.sum(J2 < -valor_critico_n2)) * 100 / escenarios }%")
log(f"J1 : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {(np.sum(J1 > valor_critico_n2) + np.sum(J1 < -valor_critico_n2)) * 100 / escenarios }%")
log("")
log(("=====  Tests No paramétricos  ====="))
log("")
log("**Cola Izquierda**")
log(f"ZR : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(ZR > valor_critico_n) * 100 / escenarios }%")
log(f"GS : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(GS > valor_critico_n) * 100 / escenarios }%")
log("**Cola derecha**")
log(f"ZR : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(ZR < -valor_critico_n) * 100 / escenarios }%")
log(f"GS : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {np.sum(GS < -valor_critico_n) * 100 / escenarios }%")
log("**A 2 colas**")
log(f"ZR : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {(np.sum(ZR > valor_critico_n2) + np.sum(ZR < -valor_critico_n2)) * 100 / escenarios }%")
log(f"GS : La probabilidad de rechazo (debajo del {nivel_confianza * 100}%) para los {escenarios} escenarios es {(np.sum(GS > valor_critico_n2) + np.sum(GS < -valor_critico_n2)) * 100 / escenarios }%")

#log("J1: ")
#log(", ".join([Fore.RED + str(round(elem, 4)) + Style.RESET_ALL if elem < nivel_confianza else str(round(elem, 4)) for elem in J1]))
#log("J2: ")
#log(", ".join([Fore.RED + str(round(elem, 4)) + Style.RESET_ALL if elem < nivel_confianza else str(round(elem, 4)) for elem in J2]))
#log("ZR: ")
#log(", ".join([Fore.RED + str(round(elem, 4)) + Style.RESET_ALL if elem < nivel_confianza else str(round(elem, 4)) for elem in ZR]))
#log("GS: ")
#log(", ".join([Fore.RED + str(round(elem, 4)) + Style.RESET_ALL if elem < nivel_confianza else str(round(elem, 4)) for elem in GS]))

# Aclaraciones sobre el trabajo en general. El estudio individual muere en el SCAR, se puede aceptar o rechazar la H0
# tomando si el SCAR cae por debajo, o por encima, de cierto valor de probabilidad de una distribucion t de student.
# Estos J1 y J2, si son distinto de 0, ya que son acumulados across the events, nos permitiran afirmar, que cuando ocurre
# determinado evento (es importante identificar correctamente el evento), a una empresa le bajaran o subiran los retornos
# sistemeticamente. Es decir que teniendo multiples J1 y J2 podremos ver como funciona nuestro modelo calculando cuantas
# veces rechazamos J1 o J2 (hay evento), cuando efectivamente sabemos que en nuestro caso no lo hay.


fig, ax = plt.subplots()
# Graficar la distribución t de Student
#ax.plot(z, pdf, 'r-', lw=2, label='Distribución t de Student')
# Graficar el histograma de las muestras generadas aleatoriamente
ax.hist(J1, bins=500, density=True, histtype='stepfilled', alpha=1)
ax.legend(loc='best', frameon=False)
plt.title('Distribución n de Normal de los J1')
plt.xlabel('Valor')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.show()

fig, ax = plt.subplots()
# Graficar la distribución t de Student
#ax.plot(z, pdf, 'r-', lw=2, label='Distribución t de Student')
# Graficar el histograma de las muestras generadas aleatoriamente
ax.hist(J2, bins=500, density=True, histtype='stepfilled', alpha=1)
ax.legend(loc='best', frameon=False)
plt.title('Distribución n de Normal de los J2')
plt.xlabel('Valor')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.show()

fig, ax = plt.subplots()
# Graficar la distribución t de Student
#ax.plot(z, pdf, 'r-', lw=2, label='Distribución t de Student')
# Graficar el histograma de las muestras generadas aleatoriamente
ax.hist(ZR, bins=500, density=True, histtype='stepfilled', alpha=1)
ax.legend(loc='best', frameon=False)
plt.title('Distribución n de Normal de los ZR')
plt.xlabel('Valor')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.show()

fig, ax = plt.subplots()
# Graficar la distribución t de Student
#ax.plot(z, pdf, 'r-', lw=2, label='Distribución t de Student')
# Graficar el histograma de las muestras generadas aleatoriamente
ax.hist(GS, bins=500, density=True, histtype='stepfilled', alpha=1)
ax.legend(loc='best', frameon=False)
plt.title('Distribución n de Normal de los GS')
plt.xlabel('Valor')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.show()

        #GRAFICO
        #fig, ax1 = plt.subplots()
        #ax1.scatter(retornos_evtwindow.index,retornos_evtwindow[ticker_rnd], s=15, label=f'retornos {ticker_rnd}')
        #ax1.plot(retornos_evtwindow['Rm'], label='Market Returns', color='red', linewidth=1)
        #ax1.plot(retornos_evtwindow.index, retornos_evtwindow['abnormalrets'], color='green', linestyle=':', linewidth=2, label='Abnormal Returns')
        #plt.title(f"Estimation Window {ticker_rnd} - {tickers_df.loc[tickers_df['Ticker'] == ticker_rnd, 'longName'].values[0]}")
        #ax1.set_xlabel('Fechas')
        #ax1.set_ylabel('Retornos')
        #plt.legend()
        #plt.show()

#print(event_data)
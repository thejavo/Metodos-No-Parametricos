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
ventana_evento = 3

fecha_finestimacion = fecha_evento - timedelta(days=gap_estimacion_evento)

q_de_gammas = 2
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

#retornos = (np.log(precios / precios.shift(1))).iloc[1:]

retornos = precios.pct_change().iloc[1:]
tickers = retornos.columns.tolist()

# Encontrar el índice de la fecha del evento en el DataFrame
indice_fecha_evento = retornos.index.get_loc(fecha_evento)
indice_comienzo_estimacion = indice_fecha_evento - gap_estimacion_evento - ventana_estimacion

# Obtener la fecha correspondiente a esa posición en el índice
fecha_estimacion = retornos.index[indice_comienzo_estimacion]
# Obtener la fecha anterior usando el índice numérico
fecha_finestimacion = retornos.index[indice_fecha_evento - 1]
fecha_finestudio = retornos.index[indice_fecha_evento + ventana_evento]

del precios

gamma = vectorgamma(ventana_evento, q_de_gammas)
scar_aux, J1, J2, ZR, GS = [], [], [], [], []

for i in range(retornos.shape[1] - 1):  # range hasta la penúltima columna

    # Datos de retornos del evento.
    est_window = retornos.loc[fecha_estimacion:fecha_finestimacion, ['^GSPC', tickers[i]]].reset_index()
    evt_window = retornos.loc[fecha_evento:fecha_finestudio, ['^GSPC', tickers[i]]].reset_index()

    # Matriz X con una columna de unos (para el término constante alfa) y los retornos del mercado
    X = np.column_stack((np.ones(len(est_window)), est_window['^GSPC'].values))

    # Vector Y con los retornos del activo
    Y = np.array(est_window[tickers[i]].values).reshape(-1, 1)  # <--- lo transformamos en un vector columna

    # MARKET MODEL - estimation window (Ri = alpha + Beta.Rm)

    # Calcular los parámetros alfa y beta usando la fórmula de MCO: (X'X)^(-1) * X'Y
    # al vector de que tiene alfa y Beta lo llamaremos theta_hat
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

    est_window['Ticker'] = tickers[i]
    est_window['L'] = 1
    # Dado que tenemos el theta_hat, podemos sacar los Retornos estimados Rm_hat = alfa + Beta * Rm
    est_window['Rm_hat'] = theta_hat[0][0] + est_window['^GSPC'] * theta_hat[1][0]

    # Si ahora a nuestros Rm_hat estimados les restamos los retornos reales que ocurrieron obtendremos
    # un estimador de nuestros desperdicios e_hat, o, lo que llamamos abnormal returns
    est_window['e_hat'] = est_window[tickers[i]] - est_window['Rm_hat']

    # ahora, dado que nuestro theta_hat depende de la muestra que tomemos, entonces nuestro theta_hat
    # tambien tiene un error intrinseco y lo mediremos a traves de su varianza VAR_theta_hat. El valor
    # esperado de theta_hat, no es otro que el verdadero theta. Recordemos la definicion de varianza
    # es la esperanza de V(x) = E((x - E(x))^2) Es el promedio de las desviaciones al cuadrado entre x y su promedio
    # Ahora como theta_hat es un vector no lo puedo elevar al cuadrado. La solucion en vectores es multiplicarlo
    # por su traspuesto, y obtendremos entonces la matriz de covarianzas V(x) = E( (x-E(x)) * (x-E(x)).T )
    # Reemplacemos theta_hat en la formula anterior.  V(theta_hat) = E( (theta_hat-E(theta_hat)) * (theta_hat-E(theta_hat)).T )
    # Ahora recordemos que la E(theta_hat) = theta entonces  V(theta_hat) = E( (theta_hat - theta) * (theta_hat - theta).T )
    # De aca recordamos que theta_hat - theta = (Xi.T * X)^-1 * Xi.T * Epsilon, todo esto lo reemplazamos en la ecuacion
    # anterior y recurrimos a la propiedad de matrices que dice (A*B).T = B.T * A.T de manera que finalmente la
    # ecuacion nos queda como  V(theta_hat) = E( (Xi.T * X)^-1 * Xi.T * Epsilon * Epsilon.T * Xi * (Xi.T * Xi)^-1 )
    # Ahora, si miramos bien, lo unico que varia realmente en la ecuacion anterio es Epsilon, o sea que para el valor
    # esperado nos podemos quedar con el termino E( Epsilon * Epsilon.T) y el resto sacarlo fuera.
    # Recurrimos ahora a dos supuestos "fuertes", que es decir que las observaciones son todas independientes entre si,
    # y si esto es verdad la covarianza entre dos Errores (Epsilon) de dos observaciones cualesquiera tiene un valor
    # esperado de 0. Y la otra suposicion "fuerte" es que todos los errores son identicamente distribuidos y por tanto
    # la volatilidad de los mismos es un valor constante definido como la cov(Epsilon). Esto, traducido a nuestro caso significa,
    # que los errores a lo largo de la ventana de estimacion de una misma muestra tienen una volatilidad constante.
    # Extiendo un poco mas, Un error tiene una volatilidad determinada por su varianza en un dia determinado, pero esa
    # volatilidad no covaria con los errores de cualquiera de los otros dias. A si que de vuelta, Existe una varianza de Epsilon
    # Pero la covarianza entre los distintos epsilon es 0.
    # Si todo esto es asi, entonces nuestro termino E ( Epsilon * Epsilon.T ) = Var(Epsilon) * Matriz_identidad y si ahora
    # reemplazo esto en la ecuacion anterior veremos que podemos simplificar muchas cosas y finalmente llegaremos a:
    # VAR_theta_hat = ( X.T * X )^-1 * Var(e)
    # Nota al margen. La Var(e) determina la amplitud de los errores de la muestra, lo que queremos es una amplitud baja
    # de forma tal de mejorar nuestras estimaciones. Por otro lado la Var(x), queremos que sea lo mas grande posible, a
    # fin de mejorar tambien nuestras estimaciones. Recordar el ejemplo balistico, queremos un caños finito (Var(e)) y
    # largo (Var(x)) a fin de lograr estimaciones mas precisas.

    # ver libro econometria de tibshirani

    # Volvemos a nuestra ecuacion de VAR_theta_hat. Nos falta obtener un valor de la Var(e)
    # No conocemos la distribucion de los verdaderos errores, pero si tenemos nuestros errores estimados
    # con lo que vamos a decir que nuestro mejor estimador de la Var(e) es la Var(e_hat) y esto es igual a
    # Var(e_hat) = 1/L1-2 * Sum(e_hat^2), donde L1 es la cantidad de datos en la ventana de estimacion. Sacamos 2
    # grados de libertad, porque estamos estimando 2 datos, alfa y Beta. Esta es la manera de obtener un estimador
    # insesgado de la varianza. Es decir, nuestra regresion tiene dos variables explicativas, si tuviera mas, deberiamos
    # restarlas

    e = (est_window['e_hat'].values).reshape(-1, 1)
    DVO_e_hat = (1 / (len(est_window) - 2) * (e.T @ e))[0][0]
    # ESP_e_hat = np.mean(est_window['e_hat'])
    VAR_theta_hat = np.linalg.inv(X.T @ X) * DVO_e_hat

    # Event Window
    evt_window['Ticker'] = tickers[i]
    evt_window['L'] = 2
    evt_window['Rm'] = theta_hat[0][0] + evt_window['^GSPC'] * theta_hat[1][0]  # usamos el theta de la estimation window
    evt_window['e_hat'] = evt_window[tickers[i]] - evt_window['Rm']

    S0 = k * DVO_e_hat  # --------------->  PARA SHOCK
    evt_window['e_hat'] += S0 * np.exp(-Lambda * evt_window.index)

    #FUNCION DE SHOCK tiene la forma S0 * e^-lambda * t . lambda es la constante de atenuacion.
    # donde S0 es funcion del desvio std multiplicado por un factor k
    # S0 = k * sigma, k toma por lo general valores (0.1, 0.5, 1, 2)
    # que pasa si lambda es 0? entonces S0 se transforma en un shock permanente k * sigma.
    # De esto se concluye que cuanto mas alto sea k y mas bajo sea lambda, mas facilmente
    # se podra detectar el efecto del evento.
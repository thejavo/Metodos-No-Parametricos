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
L1 = 101
gap_estimacion_evento = 10
pre_evento = 3
L2 = 3

fecha_finestimacion = fecha_evento - timedelta(days=gap_estimacion_evento)

q_de_gammas = 3
k = 0.5       # 0,5 - 1 -  2
Lambda = 0.1  # 0,1 - 1 - 10
#--- FIN PARAMETROS DE LA APLICACION ---

nivel_confianza = 0.05

# Datos de la distribucion t de student para el test de hipotesis parametrica
grados_libertad = L1 - 2
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
indice_comienzo_estimacion = indice_fecha_evento - gap_estimacion_evento - L1

# Obtener la fecha correspondiente a esa posición en el índice
fecha_estimacion = retornos.index[indice_comienzo_estimacion]
# Obtener la fecha anterior usando el índice numérico
fecha_finestimacion = retornos.index[indice_fecha_evento - 1 - gap_estimacion_evento]
fecha_finestudio = retornos.index[indice_fecha_evento + L2 - 1]

del precios

gamma = vectorgamma(L2, q_de_gammas)
scar_aux, J1, J2, ZR, GS = [], [], [], [], []

for i in range(retornos.shape[1] - 1):  # range hasta la penúltima columna

    # Datos de retornos del evento.
    est_window = retornos.loc[fecha_estimacion:fecha_finestimacion, ['^GSPC', tickers[i]]].reset_index()
    evt_window = retornos.loc[fecha_evento:fecha_finestudio, ['^GSPC', tickers[i]]].reset_index()

    # Matriz X con una columna de unos (para el término constante alfa) y los retornos del mercado
    X = np.column_stack((np.ones(L1), est_window['^GSPC'].values))

    # Vector Y con los retornos del activo
    Y = np.array(est_window[tickers[i]].values).reshape(-1, 1)  # <--- lo transformamos en un vector columna

    # MARKET MODEL - estimation window (Ri = alpha + Beta.Rm)

    # Calcular los parámetros alfa y beta usando la fórmula de MCO: (X'X)⁻¹ * X'Y
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
    # es la esperanza de V(X) = E((X - E(X))²) Es el promedio de las desviaciones al cuadrado entre x y su promedio
    # Ahora como theta_hat es un vector no lo puedo elevar al cuadrado. La solucion en vectores es multiplicarlo
    # por su traspuesto, y obtendremos entonces la matriz de covarianzas V(x) = E( (x-E(x)) * (x-E(x')) )
    # Reemplacemos theta_hat en la formula anterior.  V(theta_hat) = E( (theta_hat-E(theta_hat)) * (theta_hat-E(theta_hat)).T )
    # Ahora recordemos que la E(theta_hat) = theta entonces  V(theta_hat) = E( (theta_hat - theta) * (theta_hat - theta)' )
    # De aca recordamos que theta_hat - theta = (X' * X)⁻¹ * X' * Epsilon, todo esto lo reemplazamos en la ecuacion
    # anterior y recurrimos a la propiedad de matrices que dice (A*B)' = B' * A' de manera que finalmente la
    # ecuacion nos queda como  V(theta_hat) = E( (X' * X)⁻¹ * X' * Epsilon * Epsilon' * X * (X' * X)⁻¹ )
    # Ahora, si miramos bien, lo unico que varia realmente en la ecuacion anterio es Epsilon, o sea que para el valor
    # esperado nos podemos quedar con el termino E( Epsilon * Epsilon') y el resto sacarlo fuera.
    # Recurrimos ahora a dos supuestos "fuertes", que es decir que las observaciones son todas independientes entre si,
    # y si esto es verdad la covarianza entre dos Errores (Epsilon) de dos observaciones cualesquiera tiene un valor
    # esperado de 0. Y la otra suposicion "fuerte" es que todos los errores son identicamente distribuidos y por tanto
    # la volatilidad de los mismos es un valor constante definido como la cov(Epsilon). Esto, traducido a nuestro caso significa,
    # que los errores a lo largo de la ventana de estimacion de una misma muestra tienen una volatilidad constante.
    # Extiendo un poco mas, Un error tiene una volatilidad determinada por su varianza en un dia determinado, pero esa
    # volatilidad no covaria con los errores de cualquiera de los otros dias. A si que de vuelta, Existe una varianza de Epsilon
    # Pero la covarianza entre los distintos epsilon es 0.
    # Si todo esto es asi, entonces nuestro termino E ( Epsilon * Epsilon' ) = Var(Epsilon) * Matriz_identidad y si ahora
    # reemplazo esto en la ecuacion anterior veremos que podemos simplificar muchas cosas y finalmente llegaremos a:
    # VAR_theta_hat = ( X' * X )⁻¹ * Var(e)
    # Nota al margen. La Var(e_hat) determina la amplitud de los errores de la muestra, lo que queremos es una amplitud baja
    # de forma tal de mejorar nuestras estimaciones. Por otro lado la Var(X), queremos que sea lo mas grande posible, a
    # fin de mejorar tambien nuestras estimaciones. Recordar el ejemplo balistico, queremos un caños finito (Var(e_hat)) y
    # largo (Var(X)) a fin de lograr estimaciones mas precisas.

    # ver libro econometria de tibshirani

    # Volvemos a nuestra ecuacion de VAR_theta_hat. Nos falta obtener un valor de la Var(e)
    # No conocemos la distribucion de los verdaderos errores, pero si tenemos nuestros errores estimados
    # con lo que vamos a decir que nuestro mejor estimador de la Var(e) es la Var(e_hat) y esto es igual a
    # Var(e_hat) = 1/L1-2 * Sum(e_hat)^2, donde L1 es la cantidad de datos en la ventana de estimacion. Sacamos 2
    # grados de libertad, porque estamos estimando 2 datos, alfa y Beta. Esta es la manera de obtener un estimador
    # insesgado de la varianza. Es decir, nuestra regresion tiene dos variables explicativas, si tuviera mas, deberiamos
    # restarlas. La intuicion detras de esto es que, nuestro promedio sera un valor mas "grande" dado que dividiremos por
    # un N mas chico, lo que finalmente nos dara un valor mayor de varianza atendiendo a que nuestras variables
    # independientes son menos (hay 2, en este caso, que dependen del resto).
    # Terminando con la ecuacion qu enos convoca, y dado que los e_hat son vectores y que no puedo elevar vectores
    # al cuadraddo, nuestra ecuacion final de la varianza de los theta sombrero queda de la siguiente forma
    # VAR_theta_hat = (x' * x) ^ -1 * (1 / L1-2) * e_hat' * e_hat

    e = est_window['e_hat'].values
    VAR_e_hat = (1 / (L1 - 2)) * (e.T @ e)
    ESP_e_hat = np.mean(est_window['e_hat'])
    VAR_theta_hat = np.linalg.inv(X.T @ X) * VAR_e_hat

    # Event Window

    # Con nuestras 4 ecuaciones (theta_hat, Rm_hat, e_hat y Var(e_hat)) nos vamos a nuestra ventana del evento
    # y ahi deberemos obtener nuestro retornos anormales. La unica diferencia, con lo que hicimos antes
    # es que nusetro e_hat_star = Ri_star - X_star * theta_hat --> este ultimo viene del calculo
    # de la ventana de estimacion. star hace referencia a la ventana del evento.
    # Hay que tener en cuenta un par de cosas, dado que "calibramos" theta_hat, este no es exactmente theta.
    # dada la ecuacion que escribimos de e_hat_star, bajo H0 podemos decir que los Retornos del instrumento (i) en la
    # ventana del evento (Ri_star) van a ser iguales a:
    # Ri_star = X_star * theta + Epsilon_star. Reemplanzando Ri_star en nuestra ecuacion e_hat_star nos queda:
    # e_hat_star = E_star + X_star * (theta - theta_hat). Donde E(psilon)_star son los verdaderos retornos
    # anormales, que desconocemos y (theta - theta_hat) nos indica que tendremos un error en la "herramienta de medicion"
    # dado que no conocemos el verdadero theta, y todo lo que tenemos es el theta_hat que logramos armar en nuestra
    # ventana de estimación.

    X_star = np.column_stack((np.ones(L2), evt_window['^GSPC'].values))

    evt_window['Ticker'] = tickers[i]
    evt_window['L'] = 2
    evt_window['Rm'] = theta_hat[0][0] + evt_window['^GSPC'] * theta_hat[1][0]  # usamos el theta de la estimation window
    evt_window['e_hat_star'] = evt_window[tickers[i]] - evt_window['Rm']
    evt_window['e_hat_star2'] = np.array(evt_window[tickers[i]]) - (X_star @ theta_hat).reshape(-1)

    # Nos va a interesar conocer entonces la E(e_hat_star) y la VAR(e_hat_star)
    # E(e_hat_star) = E(E_star) + X_star * E(theta-theta_hat). Bajo H0, la E(E_star) deberia ser 0. Es decir, el evento
    # no tiene ningun impacto. De hecho, por construccion, en la ventana de estimacion la E(e_hat) = 0, con lo que extendemos
    # esa suposicion a la ventana del evento. En el segundo termino de la ecuacion, theta es un valor, desconocido,
    # pero al ser un valor lo sacamos de la esperanza y podemos escribir entonces el termino como theta - E(theta_hat)
    # cuando probamos el termino de insesgadez determinamos que el valor esperado de theta_hat era el verdadero theta
    # con lo cual nuestro segundo termino se hace cero tambien. Concluimos entonces que, baho H0, E(e_hat_star) = 0.
    # ESTO NO ES UN DATO MENOR!!. Si H0 es cierta, el evento no tiene ningun impacto, entonces el promedio de los
    # retornos anormales en la ventana del evento deberia ser cero.
    # Vamos con la VAR(e_hat_star), dado que e_hat_star es un vector la
    # VAR(e_hat_star) = E [ (e_hat_star - E(e_hat_star)) * (e_hat_star - E(e_hat_star))' ] esto no sera otra cosa que
    # la matriz de covarianzas. Recordemos que acabamos de definir que bajo H0 los valores esperados de e_hat_star son
    # iguales a 0. E(e_hat_star) = 0. Nos queda entonces que la VAR(e_hat_star) = E(e_hat_star * e_hat_star')
    # Habiamos dicho, en el parrafo anterior que el valor de nuestros errores se podia definir como :
    # e_hat_star = E_star + X_star * (theta - theta_hat). Reemplazamos en nuestra ecuacion:
    # VAR(e_hat_star) = E([ E_star + X_star * (theta - theta_hat)] * [ E_star + X_star * (theta - theta_hat)]') =>
    # VAR(e_hat_star) = E([ E_star + X_star * (theta - theta_hat)] * [ E_star' + (theta - theta_hat)' * X_star']) => ahora distribuimos
    # VAR(e_hat_star) = E( E_star * E_star' + X_star * (theta - theta_hat) * E_star' + E_star * (theta - theta_hat)' * X_star' + X_star * (theta - theta_hat) * (theta - theta_hat)' * X_star' ) =>

    # Antes de avanzar mas revisemos el segundo termino de la ecuacion anterior [ X_star * (theta - theta_hat) * E_star' ]
    # recordemos que theta_hat lo habiamos estimado por MCO y lo escribimos como theta_hat = (X'*X)⁻¹*(X'*Y)
    # Pero, si tuvieramos la maquina de Dios, podriamos escribir Y = X * theta + E reemplazando esto en nuestra ecuacion
    # theta_hat = (X'*X)⁻¹*(X'*(X*theta+E)) con lo que reagrupando un poco podemos escribir que:
    # theta_hat = theta + (X'*X)⁻¹ * (X'*E). De aqui podemos decir que theta - theta_hat = - (X'*X)⁻¹ * (X'*E)
    # Si vamos a nuestro termino anterior nos econtramos con que :  - [ X_star * (X'*X)⁻¹ * (X'* E * E_star') ]
    # Y de todo esto nos interesa la Cov( E E_star' ) que ya habiamos dicho que era 0, segun uno de nuestros supuestos "fuertes"
    # esto hace que todo nuestro segundo termino sea 0. Y por ende, nuestro tercer termino tambien sera 0.

    # El primer término E_star * E_star' = VAR(E) * matriz_identidad(L2), Bajo H0, los errores no cambian, la varianza
    # de los errores en la ventana del evento va a ser igual a la varianza de los errores en la ventana de estimacion.

    # El ultimo termino, el valor aleatorio que estamos midiendo es (theta - theta_hat) * (theta-theta_hat)', o sea que
    # podemos reescribirlo como X_star * E[(theta - theta_hat) * (theta-theta_hat)'] * X_star' y esta esperanza, vuelve
    # a ser la VAR(theta_hat) y recordemos que esta la escribimos como:
    # Var(theta_hat) = (X' * X)⁻¹ * (1 / (L1 - 2) * (e_hat' *  e_hat))

    # Volviendo una vez mas a la ecuación VAR(e_hat_star) a la que ya le eliminamos 2 términos y le ajustamos dos nos
    # queda que:
    # VAR(e_hat_star) = Var(e_hat) * MI(L2) + X_star * (X' * X)⁻¹ * X_star' * Var(e_hat)
    # Sacando factor comun, nos queda que:

    # VAR(e_hat_star) = Var(e_hat) * ( MI(L2) + X_star * (X' * X)⁻¹ * X_star' )

    # IMPORTANTE. Algo de intuicion. Si no tuvieramos errores de medicion, la Var(theta_hat) = 0.
    # en nuestra ultima ecuacion, diremos que la parte (X' * X)⁻¹ del segundo término, tiene que ver con la Var(theta_hat)
    # de modo que si Var(theta_hat) = 0 => la Var(e_hat_star) = Var(e_hat), es decir, en nuestro mejor escenario,
    # la varianza del los abnormal returns en la ventana del evento sera igual a la varianza de los abnormal returns
    # en la ventana de estimacion. Y esto se utilizó asi durante las primeras etapas de los estudios de eventos.
    # Sin embargo, sabemos que, dado que theta_hat es una variable estimada aleatoria hay entonces un error de
    # estimación/medicion y que por tanto Var(theta_hat) no es cero.
    # Entonces, dentro de la ventana del evento la Var(e_hat_star) estará penalizada por una varianza adicional
    # dada por el término X_star * (X' * X)⁻¹ * X_star', que llamaremos factor de correccion.
    # ¿Y porque hacemos esto?. Basicamente tenemos que entender que dado que estudiamos la distribucion de los retornos
    # anormales en la ventana de estimación, y obtuvimos una medida de su volatilidad VAR(theta_hat), lo que querriamos
    # ver, es si en la ventana de estimacion el retorno anormal es los suficientemente significativo de modo que se despegue
    # del "ruido de fondo", que viene dado por la Var(theta_hat) podriamos establecer algo parecido a un ratio
    # tipo sharpe e_star_hat / sqrt(Var(theta_hat)) y ver si el numero es lo suficientemente grande como para verificar
    # que la E(e_hat_star) es distinta de cero y entonces rechazar H0.
    # Si "subestimamos" el "ruido de fondo", es decir, decimos que durante el evento la varianza es la misma, entonces
    # estaremos induciendo a rechazar mas facilmente H0, es decir a "creer" que el evento tiene impacto cuando realmente
    # no lo tiene. Error tipo 1.
    # Si volvemos a mirar el termino  X_star * (X' * X)⁻¹ * X_star', vemos que es la razón entre las matrices de
    # covarianzas de los retornos de mercado en la ventana del evento sobre la matriz de covarizas de los retornos de
    # mercado en la ventana de estimación. Recordemos que la matriz X esta compuesta por un vector de unos y un vector
    # con los retornos de mercado. Si tomamos la ventana de estimacion entonces X es una matriz de 2 filas X L1 columnas
    # y en la ventana del evento X_star sera de 2 x L2. Por tanto la multiplicacion de X*X' y X_star*X_star' daran como
    # resultado dos matrices de covarianzas de 2x2.  Tomando X*X' tendremos en la la primer posicion de la matriz L1,
    # en la posicion 1,2 y 2,1 la sumatoria de los retornos de mercado Suma(X), y en la posicion 2,2 la sumatoria de los
    # retornos del mercado al cuadrado Suma(X)². Sacamos factor comun L1 fuera de la matriz y nos queda en la
    # posicion 1,1 un 1, en la posicion 1,2 y 2,1 (1/L1)*Suma(X) que no es otra cosa que el promedio de los retornos de
    # mercado y en la posicion 2,2 (1/L1)*Suma(X²), y esto ultimo no es otra cosa que la Var(X). Para los efectos practicos
    # el promedio de los retornos de mercado los podemos aproximar a 0, mas si tomamamos retornos diarios.
    # Ahora la matriz (X_star * X_star') sera la misma pero traspuesta (la multiplicacion es al reves). con lo que tendremos
    # L2 de factor comun y 1 en la posicion 1,1 y 0 en las posiciones 1,2 y 2,1 y finalmente en 2,2 -> (1/L2)*Suma(X_star)²
    # De vuelta como teniamos la razon entre (X_star * X_star') / (X' * X) y como, dada H0, habiamos dicho que los errores
    # eran identicamente distribuios y por tanto la volatilidad permanecia constante, entonces podemos escribir finalmente
    # que ese factor de correccion puede ser aproximado por  Var(e_hat) * (L2/L1) . ¿Cuando este factor afectara poco
    # la medicion?, obviamente cuando L1 sea mucho mas grande que L2.
    #



    print(VAR_e_hat.shape)
    print((np.ones(L2).reshape(-1, 1)).shape)
    print(X_star.shape)
    print((X_star @ np.linalg.inv(X.T @ X) @ X_star.T).shape)

    VAR_e_hat_star = VAR_e_hat * (np.ones(L2).reshape(-1, 1) + X_star @ np.linalg.inv(X.T @ X) @ X_star.T)

    # Ahora, bajo H0, vamos a decir que los e_hat_star se distribuyen como una Normal multivariada de media 0 y
    # varianza VAR_e_hat_star. Vamos a plantear los primeros test de hipotesis y para ello vamos a agrupar, agregar, los
    # retornos anormales. La agregacion puede ser a lo largo de una ventana temporal o interesectando los diferentes
    # activos (across events). La idea de "agregar" o "acumular" los retornos anoramales a lo largo de una ventana temporal
    # es ver si el efecto acumulado del evento se destaca por sobre el "ruido" que esperamos se netee.
    # Estos retornos agregados temporalmente se llaman CAR (Cumulative Abnormal Returns) estos los vamos a calcular como
    # el produtco de CAR_hat = gamma' * e_hat_star, donde gamma es un vector formado por ceros y unos, y los unos estaran
    # en los dias en que queremos "acumular" retornos.

    evt_window['car_hat'] = gamma @ evt_window['e_hat'].values

    # Como vimos anteriormente la E(e_hat_star) = 0, asi que la E(CAR_hat) = gamma.T * E(e_hat_star) con lo cual tambien es
    # 0. Ahora en cuanto a la Var(CAR_hat) = E(CAR_hat * CAR_hat') = E[gamma.T * e_hat_star * e_hat_star' * gamma],
    # e_hat_star * e_hat_star' es la Var de los retornos anormales asi que

    # Var(CAR_hat) = gammma.T * Var(e_hat_star) * gamma.

    VAR_CAR_hat = gamma.T @ VAR_e_hat_star @ gamma

    # Si los e_hat_star se distribuian como una Normal multivariada de media 0 y Varianza Var(e_hat_star) entonces
    # los CAR_hat seran Normales univariados (se trata solo de un numero) con media 0 y Var gamma.T * Var(e_hat_star) * gamma
    # Vamos a dar paso entonces a los SCAR que son los Standarized Cumulative Abnormal Returns donde dada la distribucion
    # que encontramos diremos que:

    # SCAR_hat = (CAR_hat - 0) / SQRT(Var(CAR_hat))
    # SCAR_hat = gamma.T * e_hat_star / SQRT ( gamma.T * Var(e_hat_star) * gamma)

    # pero dado que el desvio standard por el que estamos dividiendo, no es el "verdadero" desvio, es uno estimado
    # encontramos que estos SCAR_hat se distribuyen como una t de student con L1-2 grados de libertad. y aqui
    # termina la primer parte de nuestro estudio de eventos. Si solo tuviesemos 1 compañia para estudiar sus retornos,
    # solo tendriamos un SCAR. Buscaremos cual es el valor critico para un 5%, 1% etc de probabilidad de cometer error
    # de tipo 1 en una t de student con L1 - 2 grados de libertad y si nuestro SCARY cae por debajo de ese valor critico
    # entonces rechazaremos H0 y diremos que el evento si tuvo impacto. Por otro lado, si probamos otros activos y no
    # rechazamos H0 el 1% de las veces (lo que seria esperable por diseño), entonces tambien podriamos rechazar H0 y
    # decir que el evento SI tuvo impacto.

    # Dado que los SCARi son t de student su varianza sera Var(SCARi) = (L1-2)/(L1-4). Recordemos que cada SCARi
    # va a estar identicamente distribuido con el resto, todos van a ser t de student (L1-2) y dado que los aviones
    # no se ponen de acuerdo para caerse todos juntos tambien seran independientes.
    # Con todos estos SCARi podemos obtener el SCAR_prom = 1/N * suma(SCARi) y si queremos estandarizar el valor SCAR_prom
    # La Var(SCAR_prom) = 1/N² * SUMA(Var(SCARi) => Var(SCAR_prom) = ! 1:28:10
    # entonces le restamos su esperanza y lo dividimos por su desvio standard sobre raiz de N (numero de eventos)
    # y obtenemos nuestro primer estadistico J, en particular J2.

    # J2 = SQRT(N) * SCARi_prom / SQRT(L1-2/L1-4) que se distribuye como una Normal(0,1)

    # Vamos ahora con el segundo estadístico paramétrico J1.
    # La diferencia fundamental es que la agregación es across events. Seguimos calculando normalmente los abnormal returns
    # para cada activo, y ahora los vamos a promediar para cada dia en la ventana del evento y luego los agregaremos via
    # gamma. Entonces, ae(across events) e_star_ae_prom = 1/N (numero de activos) * Suma(e_hat_star) para cada dia de L2
    # CAR_prom = gamma' * e_star_ae_prom (este ultimo sera un vector de L2 x 1), con lo que nuestro CAR_prom vuelve a
    # ser un escalar. Podemos entonces calcular su varianza Var(CAR_prom) = gamma' * Var(e_star_ae_prom) * gamma
    # recordar que Var(e_star_ae_prom) va a ser una matriz de L2 x L2 y por tanto
    # Var(e_star_ae_prom) = (1 / N²) * Suma(Var(e_hat_star))  y esta ultima ya la calculamos
    # Var(e_hat_star) = VAR(e_hat) * (matriz_identidad(L2) + X_star * (X' * X)⁻¹ * X_star')
    # Nos queda entonces algo como esto:

    # Var(e_star_ae_prom) = (1/N²) * Suma( VAR(e_hat) * (matriz_identidad(L2) + X_star * (X' * X)⁻¹ * X_star'))

    # y finalmente, para calcular la Var(CAR_prom)

    # Var(CAR_prom) = gamma' * ((1/N²) * Suma( VAR(e_hat) * (matriz_identidad(L2) + X_star * (X' * X)⁻¹ * X_star'))) * gamma

    # Finalmente, estandarizaremos este numero ( numero - esperanza(numero) ) / desvio_std(numero) y esto sera nuestro
    # estadistico J1.

    # J1 = (CAR_prom - 0) / SQRT(Var(CAR_prom) =>
    # J1 = gamma' * e_star_ae_prom / SQRT( gamma' * Var(e_star_ae_prom) * gamma ) =>
    # J1 = gamma' * e_star_ae_prom / SQRT( gamma' * ((1/N²) * Suma( VAR(e_hat) * (matriz_identidad(L2) + X_star * (X' * X)⁻¹ * X_star'))) * gamma )

    # y esto se distribuye como N(0,1), ya que venimos de varias agregaciones y estas responden al TCL


    S0 = k * VAR_e_hat  # --------------->  PARA SHOCK
    evt_window['e_hat'] += S0 * np.exp(-Lambda * evt_window.index)

    #FUNCION DE SHOCK tiene la forma S0 * e^-lambda * t . lambda es la constante de atenuacion.
    # donde S0 es funcion del desvio std multiplicado por un factor k
    # S0 = k * sigma, k toma por lo general valores (0.1, 0.5, 1, 2)
    # que pasa si lambda es 0? entonces S0 se transforma en un shock permanente k * sigma.
    # De esto se concluye que cuanto mas alto sea k y mas bajo sea lambda, mas facilmente
    # se podra detectar el efecto del evento.
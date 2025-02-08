import pandas as pd
import numpy as np
import scipy.stats as stats
from graficos import graficar_activo, graficar_scar, graficar_ar_promedio

def evaluar_contraste(descripcion, valor_prueba, valor_critico, colas="2"):
    """
    Evalúa un contraste de hipótesis y genera un mensaje de resultado.
    """
    if colas == "I" and valor_prueba < valor_critico:
        return f"Se rechaza H0 para {descripcion}: Hay evidencia de impacto negativo significativo."
    elif colas == "D" and valor_prueba > valor_critico:
        return f"Se rechaza H0 para {descripcion}: Hay evidencia de impacto positivo significativo."
    elif colas == "2" and abs(valor_prueba) > valor_critico:
        return f"Se rechaza H0 para {descripcion}: Hay evidencia de impacto significativo."
    return f"No se rechaza H0 para {descripcion}: No hay evidencia de impacto significativo."


# --- CONFIGURACIÓN DE PARÁMETROS ---
nombre_evento = "Estatización de Vicentin"
fecha_evento = pd.to_datetime("2020-06-08")  # Modificable según evento
gap_estimacion_evento = 10  # Gap entre ventana de estimación y evento
L1 = 100  # Ventana de estimación (en días bursátiles)

dias_previos = 0  # para calcular gamma
dias_posteriores = 4  # para calcular gamma.
L2 = dias_previos + dias_posteriores + 1  # Ventana de evento

market_benchmark = "^GSPC"

# Configuración inicial del vector gamma
qdeunos = 3
cerosantes = 0
cerosdesp = L2 - (qdeunos + cerosantes)
if cerosdesp < 0:
    raise ValueError("el tamaño del vector gamma no puede ser mayor a L2")

gamma = np.pad(np.ones(qdeunos), (cerosantes, cerosdesp), mode='constant')

# --- LECTURA DE DATOS ---
precios = pd.read_csv("precios.csv", index_col="Date", parse_dates=["Date"])
precios.index = precios.index.tz_localize(None)

if market_benchmark not in precios.columns:
    raise ValueError(f"El benchmark '{market_benchmark}' no está presente en el archivo.")

retornos = precios.pct_change().iloc[1:]

tickers = [col for col in retornos.columns if col != market_benchmark]

# --- CÁLCULO DE VENTANAS ---
indice_fecha_evento = retornos.index.get_loc(fecha_evento)
fecha_fin_estimacion = retornos.index[indice_fecha_evento - gap_estimacion_evento - 1]
fecha_inicio_estimacion = retornos.index[indice_fecha_evento - gap_estimacion_evento - L1]
fecha_inicio_evento = retornos.index[indice_fecha_evento - dias_previos]
fecha_fin_evento = retornos.index[indice_fecha_evento + dias_posteriores]

# --- INICIALIZACIÓN DE RESULTADOS ---
resultados = {
    'theta_hat': {},  # Almacena alfa y beta de cada activo
    'car': {},  # Retornos acumulados anormales
    'scar': {},  # Retornos acumulados anormales estandarizados
    'var_car': {}  # Varianza de los CAR
}

AR_star_matriz = np.zeros((len(tickers), L2))
# Crear un array tridimensional para almacenar las suma de matrices var_AR_star
suma_var_AR_star = np.zeros((L2, L2))

# --- MODELO DE MERCADO Y CÁLCULO DE ESTADÍSTICOS ---
i = 0 #indice para matrices auxiliares
for ticker in tickers:

    if retornos.loc[fecha_inicio_estimacion:fecha_fin_estimacion, ticker].isna().any() or \
            retornos.loc[fecha_inicio_evento:fecha_fin_evento, ticker].isna().any():
        print(f"Activo {ticker} descartado por valores N/A")
        continue #este 'continue' salta el bucle completo del for, por lo que no es necesario un else a continuacion.

    # Ventanas de estimación y evento
    est_window = retornos.loc[fecha_inicio_estimacion:fecha_fin_estimacion, [market_benchmark, ticker]]
    evt_window = retornos.loc[fecha_inicio_evento:fecha_fin_evento, [market_benchmark, ticker]]

    # Matriz X y vector Y para el modelo de mercado
    X_hat = np.column_stack((np.ones(L1), est_window[market_benchmark].values))
    Y_hat = est_window[ticker].values.reshape(-1, 1)

    # Estimación de parámetros alfa y beta
    theta_hat = (np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ Y_hat).flatten()
    resultados['theta_hat'][ticker] = theta_hat.tolist()

    # Cálculo de retornos anormales (AR)
    est_window['AR'] = est_window[ticker] - (theta_hat[0] + theta_hat[1] * est_window[market_benchmark])
    evt_window['AR'] = evt_window[ticker] - (theta_hat[0] + theta_hat[1] * evt_window[market_benchmark])

    resultados.setdefault('ar', {})[ticker] = {
        'estimacion': est_window['AR'].values.tolist(),
        'evento': evt_window['AR'].values.tolist()
    }

    AR_star_matriz[i, :] = evt_window['AR'].values
    i += 1

    # Cálculo de varianza de AR en la ventana de estimación
    e = est_window['AR'].values
    var_AR_est = (1 / (L1 - 2)) * (e.T @ e)

    # Matriz X* para la ventana de evento
    X_star = np.column_stack((np.ones(L2), evt_window[market_benchmark].values))

    # Varianza AR en la ventana de evento
    var_AR_star =  var_AR_est * (np.eye(L2) + X_star @ np.linalg.inv(X_hat.T @ X_hat) @ X_star.T)
    suma_var_AR_star += var_AR_star

    # Cálculo de varianza de CAR
    var_CAR = gamma.T @ var_AR_star @ gamma
    resultados['var_car'][ticker] = var_CAR.item()

    # Cálculo de CAR (Cumulative Abnormal Returns)
    CAR = evt_window['AR'].values @ gamma.T
    resultados['car'][ticker] = CAR

    # Cálculo de SCAR (Standardized Cumulative Abnormal Returns)
    SCAR = CAR / np.sqrt(var_CAR)
    resultados['scar'][ticker] = SCAR

# --- CÁLCULO DE ESTADÍSTICOS J1 Y J2 ---
SCAR_values = np.array(list(resultados['scar'].values()))

# Desvio teórico de los SCAR (t de student)
std_SCAR = np.sqrt((L1 - 2) / (L2 - 4))
n = len(SCAR_values)

# J2: Promedio estandarizado de SCAR dividido por su varianza (ajustado para t de Student)
J2 = np.sqrt(n) * SCAR_values.mean() / std_SCAR

# J1: CAR promedio diario across events ajustado con la desviación estándar teórica
AR_prom_diario_across_events = np.mean(AR_star_matriz, axis=0)
var_AR_prom_diario = suma_var_AR_star / n**2
CAR_prom_diario = gamma.T @ AR_prom_diario_across_events
var_CAR_prom_diario = gamma.T @ var_AR_prom_diario @ gamma

J1 = CAR_prom_diario / np.sqrt(var_CAR_prom_diario)

print ("==== Tests PARAMETRICOS J1 y J2 ====")
print (f"J1 = {J1}")
print (f"J2 = {J2}")
print()

# --- CONTRASTE CONTRA H0 ---
# H0: El evento no tiene impacto significativo
nivel_confianza = 0.95
alfa = 1 - nivel_confianza #nivel de significancia
df = L1 - 2  #grados de libertad

critico_t_2colas = stats.t.ppf(1 - alfa / 2, df=df)
critico_t_1cola = stats.t.ppf(alfa, df=df)

print(f"Valor crítico para t (dos colas): ±{critico_t_2colas}")
print(f"Valor crítico para t cola izquierda: {critico_t_1cola}")

print(evaluar_contraste("J1 (dos colas)", J1, critico_t_2colas))
print(evaluar_contraste("J1 (una cola)", J1, critico_t_1cola,"I"))

print(evaluar_contraste("J2 (dos colas)", J2, critico_t_2colas))
print(evaluar_contraste("J2 (una cola)", J2, critico_t_1cola,"I"))
print()

# --- TEST NO PARAMÉTRICOS ---
# Test de Signos
signos_prom = []
signos_dia_evento = []
for ticker in tickers:
    signos = (np.array(resultados['ar'][ticker]['estimacion']) > 0).astype(int)
    signo_prom_ticker = np.sum(signos) / L1
    signos_prom.append(signo_prom_ticker)

    AR_evento_dia = resultados['ar'][ticker]['evento'][dias_previos]  # El día del evento es el índice dias_previos
    signo = 1 if resultados['ar'][ticker]['evento'][dias_previos] > 0 else 0
    signos_dia_evento.append(signo)

p_est = np.mean(signos_prom)
X = np.sum(signos_dia_evento)
N = len(tickers)
GS = (X - N * p_est) / np.sqrt(N*p_est*(1-p_est))
critico_bin_1cola = stats.binom.ppf(1 - alfa, N, p_est)

print ("====    Tests NO PARAMETRICOS    ====")
print ("==== Test de signos generalizado ====")
print (f"GS = {GS}")
print (f"p_est = {p_est}")
print (f"Valor critico binomial cola derecha = {critico_bin_1cola}")
print(evaluar_contraste("GS (una cola)", GS, critico_bin_1cola,"D"))
print()


# Test de Rangos
rankings = np.zeros((len(tickers), L1 + L2))
for i, ticker in enumerate(tickers):
    rankings[i, :] = stats.rankdata(np.concatenate((resultados['ar'][ticker]['estimacion'],  resultados['ar'][ticker]['evento'])))

rank_medio = int((L1 + L2) / 2)
ranks_promedios_diarios = np.mean(rankings, axis=0)
dstd_ranking = np.sqrt(np.sum((ranks_promedios_diarios - rank_medio) ** 2) / len(tickers))

ranking_prom_evt = np.mean(rankings[:, -L2:]) # tomo todas las filas y las ultimas L2 columnas de la matriz de rankings y
                                              # al no especificar axis= en el np.mean python aplana la matriz y promedia,
                                              # que es lo mismo que promediar por activo cada dia y luego volver a promediar.

ZR = np.sqrt(L2) * (ranking_prom_evt - rank_medio) / dstd_ranking

# Valores críticos para N(0, 1)
critico_N_2colas = stats.norm.ppf(1 - alfa / 2)
critico_N_1cola = stats.norm.ppf(alfa)

print("==== Test de Ranking ====")
print(f"ZR = {ZR}")
print(f"Valor critico normal 2 colas = ±{critico_N_2colas}")
print(f"Valor critico normal cola izquierda = {critico_N_1cola}")
print(evaluar_contraste("Test de Ranking (2 colas)", ZR, critico_N_2colas))
print(evaluar_contraste("Test de Ranking (1 cola)", ZR, critico_N_1cola, "I"))
print()

# --- RESULTADOS ---
print("Resultados finales:")
print(resultados)

graficar_ar_promedio(nombre_evento, AR_prom_diario_across_events, dias_previos, dias_posteriores)
graficar_scar(resultados,critico_t_2colas)

# --- EVALUAR TICKERS EN FORMA INDIVIDUAL PARA GRAFICAR SI LOS CONSIDERAMOS SIGNIFICATIVOS ---
umbral_varianza = 0.001
umbral_diferencia = 0.01

car_benchmark = np.mean(list(resultados['car'].values()))  # Benchmark del CAR

for ticker in tickers:

    if resultados['var_car'][ticker] < umbral_varianza or \
       abs(resultados['car'][ticker] - car_benchmark) < umbral_diferencia or \
       abs(resultados['scar'][ticker]) < critico_t_2colas:
        continue

    print(f"Activo {ticker} es representativo de un posible evento.")

    # Graficar solo si es representativo
    graficar_activo(
        ticker,
        retornos.loc[fecha_inicio_estimacion:fecha_fin_estimacion, [market_benchmark, ticker]],
        retornos.loc[fecha_inicio_evento:fecha_fin_evento, [market_benchmark, ticker]],
        resultados['theta_hat'][ticker],
        fecha_inicio_evento,
        fecha_fin_evento
    )

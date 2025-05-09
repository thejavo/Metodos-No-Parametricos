import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from estudio_eventos import estudio_eventos

# Parámetros generales
N = 500  # Número de iteraciones
L1 = 100
gap_estimacion_evento = 10  # Excluir algunas fechas iniciales y finales para evitar bordes
qdeunos = 3
meses_hacia_atras = 12
datafile = "CierresMERVAL.csv"
marketbenchmark = "^MERV"
fecha_evento_real = pd.Timestamp("2024-07-11") #2020-06-08 y 2024-07-11

# 📌 Leer precios y asegurar formato correcto
precios = pd.read_csv(datafile, index_col="Date", parse_dates=["Date"])
precios.index = precios.index.tz_localize(None)  # Eliminar timezone si existe
precios = precios.dropna(subset=[marketbenchmark])  # Asegurar que el benchmark no tenga NaN

# Calculo de fechas para la simulacion
fecha_max_datos = precios.index[-qdeunos] # Última fecha válida sin pisar el gap
fecha_max = min(fecha_evento_real, fecha_max_datos) #fecha máxima es la más restrictiva entre ambas

# 📌 Determinar fecha mínima basada en horizonte temporal reciente
fecha_min = fecha_evento_real - pd.DateOffset(months=meses_hacia_atras)
fecha_min = max(fecha_min, precios.index[L1 + gap_estimacion_evento])

# 📌 Filtrar fechas dentro del rango permitido
fechas_disponibles = precios.loc[fecha_min:fecha_max].index

# 📌 Verificar si hay suficientes fechas para la simulación
#if len(fechas_disponibles) < N:
#    raise ValueError(f"No hay suficientes fechas disponibles ({len(fechas_disponibles)}) para {N} simulaciones.")

print(f"{len(fechas_disponibles)} fechas disponibles para {N} simulaciones con reposicion")

# 📌 Seleccionar fechas aleatorias
fechas_simuladas = pd.to_datetime(np.random.choice(fechas_disponibles, N, replace=True))

# 📌 Listas para almacenar resultados de las simulaciones
j1_poblacion = []
j2_poblacion = []
bmp_poblacion = []

for i, fecha in enumerate(fechas_simuladas):
#for fecha in fechas_simuladas:
  try:
    print(f"\rSimulación {i + 1}/{N} - Fecha: {fecha.date()} ", end="")
    resultado = estudio_eventos(
        fecha_evento=fecha,
        L1=L1,
        indice_dia_evento=3,
        dias_previos=3,
        dias_posteriores=4,
        filename=datafile,
        nombre_evento="Simulación",
        gap_estimacion_evento=gap_estimacion_evento,
        market_benchmark=marketbenchmark,
        nivel_confianza=0.95,
        qdeunos=qdeunos,
        cerosantes=0,
        omitir_tickers=['HGLD', 'VIST']
    )

    # Guardar valores en listas para construir histogramas
    j1_poblacion.append(resultado["J1"])
    j2_poblacion.append(resultado["J2"])
    bmp_poblacion.append(resultado["BMP"])

  except Exception as e:
    print(f"⚠️  Error en simulación para la fecha {fecha}: {e}")
    N = N - 1
    continue

print("\n")

#fecha_evento_real = pd.Timestamp("2024-07-11") #2020-06-08
resultado_real = estudio_eventos(
    fecha_evento=fecha_evento_real,
    L1=100,
    indice_dia_evento=3,
    dias_previos=3,
    dias_posteriores=4,
    filename=datafile,
    nombre_evento="Evento Real",
    gap_estimacion_evento=gap_estimacion_evento,
    market_benchmark=marketbenchmark,
    nivel_confianza=0.95,
    qdeunos=3,
    cerosantes=0,
    omitir_tickers=['HGLD', 'VIST']
)

# Extraer valores reales
j1_real = resultado_real["J1"]
j2_real = resultado_real["J2"]
bmp_real = resultado_real["BMP"]

print(f"{len(fechas_disponibles)} fechas disponibles para {N} simulaciones con reposicion")

resumen = {
    'Estadístico': ['J1', 'J2', 'BMP'],
    'Valor observado': [j1_real, j2_real, bmp_real],
    'Media simulada': [np.mean(j1_poblacion), np.mean(j2_poblacion), np.mean(bmp_poblacion)],
    'Percentil 2.5%': [np.percentile(j1_poblacion, 2.5), np.percentile(j2_poblacion, 2.5), np.percentile(bmp_poblacion, 2.5)],
    'Percentil 97.5%': [np.percentile(j1_poblacion, 97.5), np.percentile(j2_poblacion, 97.5), np.percentile(bmp_poblacion, 97.5)],
    'p-valor empírico': [
        np.mean(np.abs(j1_poblacion) >= abs(j1_real)),
        np.mean(np.abs(j2_poblacion) >= abs(j2_real)),
        np.mean(np.abs(bmp_poblacion) >= abs(bmp_real))
    ]
}

tabla = pd.DataFrame(resumen)
print(tabla.to_string(index=False))

# 📌 Crear histogramas para cada población con la línea del evento real
plt.figure(figsize=(12, 4))

# Histograma de J1
plt.subplot(1, 3, 1)
plt.hist(j1_poblacion, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(j1_real, color='black', linestyle='dashed', linewidth=2, label='Evento real')
plt.xlabel("J1")
plt.ylabel("Frecuencia")
plt.title("Distribución de J1")
plt.legend()

# Histograma de J2
plt.subplot(1, 3, 2)
plt.hist(j2_poblacion, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.axvline(j2_real, color='black', linestyle='dashed', linewidth=2, label='Evento real')
plt.xlabel("J2")
plt.title("Distribución de J2")
plt.legend()

# Histograma de BMP
plt.subplot(1, 3, 3)
plt.hist(bmp_poblacion, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.axvline(bmp_real, color='black', linestyle='dashed', linewidth=2, label='Evento real')
plt.xlabel("BMP")
plt.title("Distribución de BMP")
plt.legend()

plt.tight_layout()
plt.show()

print(f"✅ Simulación completada. Evento real (2020-06-08): J1={j1_real:.4f}, J2={j2_real:.4f}, BMP={bmp_real:.4f}")



"""
# 📌 Crear histogramas para cada población
plt.figure(figsize=(12, 4))

# Histograma de J1
plt.subplot(1, 3, 1)
plt.hist(j1_poblacion, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("J1")
plt.ylabel("Frecuencia")
plt.title("Distribución de J1")

# Histograma de J2
plt.subplot(1, 3, 2)
plt.hist(j2_poblacion, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.xlabel("J2")
plt.title("Distribución de J2")

# Histograma de BMP
plt.subplot(1, 3, 3)
plt.hist(bmp_poblacion, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.xlabel("BMP")
plt.title("Distribución de BMP")

plt.tight_layout()
plt.show()
"""
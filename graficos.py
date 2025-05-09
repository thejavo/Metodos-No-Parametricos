import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def graficar_MarketModel(R_m, R_i, alpha, beta, ticker):
    """
    Grafica la nube de puntos (Rm, Ri), la recta de regresión, y marca alfa en el eje Y.

    Parámetros:
    R_m -- Serie de retornos del mercado
    R_i -- Serie de retornos del activo
    alpha -- Intercepto de la regresión
    beta -- Pendiente de la regresión
    ticker -- Nombre del activo
    """
    plt.figure(figsize=(10, 6))

    # Graficar la nube de puntos
    plt.scatter(R_m, R_i, label="Datos (Rm, Ri)", alpha=0.6, color="blue")

    # Crear la recta de regresión
    Rm_min, Rm_max = R_m.min(), R_m.max()
    Ri_min = alpha + beta * Rm_min
    Ri_max = alpha + beta * Rm_max
    plt.plot([Rm_min, Rm_max], [Ri_min, Ri_max], color="red", linewidth=2, label="Recta MCO")

    # Línea horizontal para alfa
    plt.plot([Rm_min, 0], [alpha, alpha], color="green", linestyle="--", linewidth=1, label="$\\alpha$")

    # Mostrar el valor de alfa cerca del eje Y
    plt.text(
        Rm_min + 0.01, alpha, f"$\\alpha$ = {alpha:.4f}",
        fontsize=10, color="green", va="bottom", ha="left",
        transform_rotates_text=True, rotation=0, rotation_mode='anchor'
    )

    # Escribir beta sobre la línea roja en el ángulo correspondiente
    angle = np.rad2deg(np.arctan(beta))
    plt.text(
        Rm_min + 0.01, Ri_min + 0.01, f"$\\beta$ = {beta:.4f}",
        fontsize=10, color="red", ha="left", va="bottom",
        transform_rotates_text=True, rotation=angle, rotation_mode='anchor'
    )

    #--- Recta de ejemplo de Et

    # Calcular las distancias entre cada punto y la recta OLS
    distancias = np.abs(R_i - (alpha + beta * R_m))

    # Encontrar el índice del punto con la mayor distancia
    indice_max_distancia = np.argmax(distancias)

    # Coordenadas del punto más alejado
    Rm_ejemplo = R_m.iloc[indice_max_distancia]
    Ri_ejemplo = R_i.iloc[indice_max_distancia]

    # Calcular el valor en la recta OLS correspondiente a Rm_ejemplo
    Ri_recta = alpha + beta * Rm_ejemplo

    # Dibujar la línea vertical entre el punto y la recta OLS
    plt.plot([Rm_ejemplo, Rm_ejemplo], [Ri_recta, Ri_ejemplo], color="orange", linestyle="--", linewidth=1.5,  label="$\\varepsilon_t$")

    # Etiqueta para Et
    plt.text(
        Rm_ejemplo, Ri_ejemplo+0.01,  # Posición de la etiqueta en el medio de la línea
        r"Ejemplo de $\varepsilon_t$", fontsize=10, color="orange", ha="left", va="bottom",
        transform_rotates_text=True, rotation=90, rotation_mode='anchor'
    )
    # --- Fin recta de ejemplo de Et

    # Línea vertical en X = 0
    plt.axvline(x=0, color="gray", linestyle="--", linewidth=1.5)

    # Ajustar los límites del eje X para garantizar que el eje Y esté incluido
    plt.xlim(left=min(Rm_min, 0))  # Asegurar que x=0 esté incluido

    # Etiquetas y formato
    plt.xlabel("Retorno del Mercado (Rm)")
    plt.ylabel(f"Retorno del Activo ({ticker})")
    plt.title(f"Nube de Puntos y Regresión para {ticker}")

    # Ajustar la posición de la leyenda a la esquina inferior derecha
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def graficar_Homocedasticidad(R_m, R_i, alpha, beta, ticker):
    """
    Gráfico de residuos (ε_t) vs. R_m para evaluar la homocedasticidad,
    incluyendo un área de ±3σ.

    Parámetros:
    R_m -- Serie de retornos del mercado
    R_i -- Serie de retornos del activo
    alpha -- Intercepto de la regresión
    beta -- Pendiente de la regresión
    ticker -- Nombre del activo
    """
    # Calcular los residuos
    epsilon_t = R_i - (alpha + beta * R_m)

    # Calcular el desvío estándar
    sigma = np.std(epsilon_t)
    rango_superior = sigma * 2
    rango_inferior = -sigma * 2

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Área sombreada para ±3σ
    plt.fill_between(
        np.linspace(R_m.min(), R_m.max(), 500),
        rango_inferior,
        rango_superior,
        color="lightgray",
        alpha=0.3,
        label=r"Área de $\pm2\sigma$"
    )

    # Línea horizontal en y=0 para referencia
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label=r"Referencia ($\epsilon_t = 0$)")

    # Graficar los puntos de los residuos
    plt.scatter(R_m, epsilon_t, alpha=0.6, color="blue", label=r"Residuos ($\epsilon_t$)")

    # Ajustar los límites del eje X para que el area sombreada vaya de borde a borde
    plt.xlim(left=min(R_m.min(), 0), right=max(R_m.max(),0))

    # Etiquetas y título
    plt.xlabel("Retorno del Mercado ($R_m$)")
    plt.ylabel(r"Residuos ($\epsilon_t$)")
    plt.title(f"Gráfico de Homocedasticidad para {ticker}\n(Área sombreada ±2σ: {sigma * 2:.4f})")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

def graficar_epsilons(residuos, ticker):
    """
    Genera un gráfico de los residuos (epsilon_t) en función del tiempo para observar posibles patrones de autocorrelación.

    Parámetros:
    residuos -- Array o lista de residuos (epsilon_t) en orden temporal.
    ticker -- Nombre del activo (para el título del gráfico).
    """
    plt.figure(figsize=(10, 6))

    # Graficar la evolución de los residuos en el tiempo
    plt.plot(residuos, marker='o', linestyle='-', color='blue', alpha=0.7, label=r"Residuos ($\epsilon_t$)")
    #plt.scatter(R_m, epsilon_t, alpha=0.6, color="blue", label=r"Residuos ($\epsilon_t$)")

    # Línea horizontal de referencia en y=0
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label=r"Referencia ($\epsilon_t = 0$)")

    # Etiquetas y formato
    plt.xlabel("Tiempo (t)")
    plt.ylabel(r"Residuos ($\epsilon_t$)")
    plt.title(r"Evolución de los Residuos ($\epsilon_t$)"+f" para {ticker}\n¿Hay Autocorrelación?")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    # Mostrar el gráfico
    plt.show()

def graficar_activo(ticker, est_window, evt_window, theta_hat, fecha_inicio_evento, fecha_fin_evento):
    """
    Genera un gráfico para un activo mostrando retornos estimados y efectivos en las ventanas de estimación y evento.

    Args:
        ticker (str): Nombre del activo.
        est_window (pd.DataFrame): Ventana de estimación.
        evt_window (pd.DataFrame): Ventana de evento.
        theta_hat (np.ndarray): Coeficientes alfa y beta del modelo de mercado.
        fecha_inicio_evento (pd.Timestamp): Fecha inicial de la ventana de evento.
        fecha_fin_evento (pd.Timestamp): Fecha final de la ventana de evento.
    """
    # Retornos estimados y anormales en la ventana de estimación
    est_window['R_est'] = theta_hat[0] + theta_hat[1] * est_window['^GSPC']
    est_window['AR'] = est_window[ticker] - est_window['R_est']

    # Retornos estimados y anormales en la ventana de evento
    evt_window['R_est'] = theta_hat[0] + theta_hat[1] * evt_window['^GSPC']
    evt_window['AR'] = evt_window[ticker] - evt_window['R_est']

    # Combinar datos para el gráfico
    est_window['Ventana'] = 'Estimación'
    evt_window['Ventana'] = 'Evento'
    combined_data = pd.concat([est_window, evt_window])
    combined_data['Fecha'] = combined_data.index

    # Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data['Fecha'], combined_data[ticker], label='R_i (Efectivo)', color='blue')
    plt.plot(combined_data['Fecha'], combined_data['R_est'], label='R_i (Estimado)', color='red', linestyle='--')
    plt.bar(combined_data['Fecha'], combined_data['AR'], label='AR (Retornos Anormales)', color='green', alpha=0.4)

    # Resaltar la ventana de evento
    plt.axvspan(fecha_inicio_evento, fecha_fin_evento, color='yellow', alpha=0.2, label='Ventana de Evento')

    # Configurar el gráfico
    plt.title(f'Retornos del Activo {ticker}')
    plt.xlabel('Fecha')
    plt.ylabel('Retornos')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def graficar_ar_promedio(evento, ar_promedio, dias_previos, dias_posteriores):
    """
    Grafica el CAR promedio en torno al evento.

    Args:
        resultados (dict): Diccionario con los CAR de todos los activos.
        dias_previos (int): Número de días previos al evento.
        dias_posteriores (int): Número de días posteriores al evento.
    """
    dias_evento = range(-dias_previos, dias_posteriores + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(dias_evento, ar_promedio, label='AR Promedio', color='blue')
    plt.axvline(0, color='red', linestyle='--', label='Día del Evento')
    plt.title('AR Promedio en Torno al Evento'+' '+evento)
    plt.xlabel('Días Relativos al Evento')
    plt.ylabel('AR Promedio')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_scar(resultados, critico):
    """
    Grafica un histograma de los SCAR de todos los activos.

    Args:
        resultados (dict): Diccionario con los SCAR de todos los activos.
    """
    scar_values = list(resultados['scar'].values())

    plt.figure(figsize=(10, 6))
    plt.hist(scar_values, bins=10, color='green', alpha=0.7, label='SCAR')
    plt.axvline(critico, color='orange', linestyle='--', label='Confianza 95% (Positiva)')
    plt.axvline(-critico, color='orange', linestyle='--', label='Confianza 95% (Negativa)')
    plt.title('Distribución de los SCAR')
    plt.xlabel('SCAR')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
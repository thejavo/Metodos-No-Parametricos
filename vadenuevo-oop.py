import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, file_path, market_benchmark):
        self.file_path = file_path
        self.market_benchmark = market_benchmark

    def load_data(self):
        precios = pd.read_csv(self.file_path, index_col="Date", parse_dates=["Date"])
        precios.index = precios.index.tz_localize(None)
        if self.market_benchmark not in precios.columns:
            raise ValueError(f"El benchmark '{self.market_benchmark}' no está presente en el archivo.")
        retornos = precios.pct_change().iloc[1:]
        return retornos

class MarketModel:
    def __init__(self, est_window, evt_window, market_benchmark, ticker):
        self.est_window = est_window
        self.evt_window = evt_window
        self.market_benchmark = market_benchmark
        self.ticker = ticker

    def calculate_theta_and_ar(self):
        X_hat = np.column_stack((np.ones(len(self.est_window)), self.est_window[self.market_benchmark].values))
        Y_hat = self.est_window[self.ticker].values.reshape(-1, 1)
        self.theta_hat = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ Y_hat

        self.est_window['AR'] = self.est_window[self.ticker] - (self.theta_hat[0][0] + self.theta_hat[1][0] * self.est_window[self.market_benchmark])
        self.evt_window['AR'] = self.evt_window[self.ticker] - (self.theta_hat[0][0] + self.theta_hat[1][0] * self.evt_window[self.market_benchmark])

        return self.theta_hat, self.est_window, self.evt_window

class StatisticsCalculator:
    def __init__(self, est_window, evt_window, theta_hat, gamma):
        self.est_window = est_window
        self.evt_window = evt_window
        self.theta_hat = theta_hat
        self.gamma = gamma

    def calculate_varianza_ar(self):
        e = self.est_window['AR'].values
        var_AR_est = (1 / (len(self.est_window) - 2)) * (e.T @ e)

        X_star = np.column_stack((np.ones(len(self.evt_window)), self.evt_window[self.market_benchmark].values))
        var_AR_star = var_AR_est * (np.eye(len(self.evt_window)) + X_star @ np.linalg.inv(X_hat.T @ X_hat) @ X_star.T)

        return var_AR_star

    def calculate_car_and_scar(self, var_AR_star):
        var_CAR = self.gamma.T @ var_AR_star @ self.gamma
        CAR = self.evt_window['AR'].values @ self.gamma.T
        SCAR = CAR / np.sqrt(var_CAR)
        return CAR, SCAR, var_CAR

class HypothesisTester:
    def __init__(self, alfa, df):
        self.alfa = alfa
        self.df = df

    def evaluate_contrast(self, descripcion, valor_prueba, valor_critico, colas="2"):
        if colas == "I" and valor_prueba < valor_critico:
            return f"Se rechaza H0 para {descripcion}: Hay evidencia de impacto negativo significativo."
        elif colas == "D" and valor_prueba > valor_critico:
            return f"Se rechaza H0 para {descripcion}: Hay evidencia de impacto positivo significativo."
        elif colas == "2" and abs(valor_prueba) > valor_critico:
            return f"Se rechaza H0 para {descripcion}: Hay evidencia de impacto significativo."
        return f"No se rechaza H0 para {descripcion}: No hay evidencia de impacto significativo."

class Plotter:
    @staticmethod
    def graficar_activo(ticker, est_window, evt_window, theta_hat, fecha_inicio_evento, fecha_fin_evento):
        est_window['R_est'] = theta_hat[0][0] + theta_hat[1][0] * est_window['^GSPC']
        est_window['AR'] = est_window[ticker] - est_window['R_est']
        evt_window['R_est'] = theta_hat[0][0] + theta_hat[1][0] * evt_window['^GSPC']
        evt_window['AR'] = evt_window[ticker] - evt_window['R_est']

        combined_data = pd.concat([est_window, evt_window])
        combined_data['Fecha'] = combined_data.index

        plt.figure(figsize=(12, 6))
        plt.plot(combined_data['Fecha'], combined_data[ticker], label='R_i (Efectivo)', color='blue')
        plt.plot(combined_data['Fecha'], combined_data['R_est'], label='R_i (Estimado)', color='red', linestyle='--')
        plt.bar(combined_data['Fecha'], combined_data['AR'], label='AR (Retornos Anormales)', color='green', alpha=0.4)
        plt.axvspan(fecha_inicio_evento, fecha_fin_evento, color='yellow', alpha=0.2, label='Ventana de Evento')
        plt.title(f'Retornos del Activo {ticker}')
        plt.xlabel('Fecha')
        plt.ylabel('Retornos')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

class EventStudy:
    def __init__(self, file_path, market_benchmark, fecha_evento, gap_estimacion_evento, L1, dias_previos, dias_posteriores):
        self.file_path = file_path
        self.market_benchmark = market_benchmark
        self.fecha_evento = fecha_evento
        self.gap_estimacion_evento = gap_estimacion_evento
        self.L1 = L1
        self.dias_previos = dias_previos
        self.dias_posteriores = dias_posteriores
        self.L2 = dias_previos + dias_posteriores + 1
        self.gamma = self._create_gamma_vector()

    def _create_gamma_vector(self):
        qdeunos = 3
        cerosantes = 4
        cerosdesp = self.L2 - (qdeunos + cerosantes)
        if cerosdesp < 0:
            raise ValueError("El tamaño del vector gamma no puede ser mayor a L2")
        return np.pad(np.ones(qdeunos), (cerosantes, cerosdesp), mode='constant')

    def run(self):
        data_loader = DataLoader(self.file_path, self.market_benchmark)
        retornos = data_loader.load_data()

        indice_fecha_evento = retornos.index.get_loc(self.fecha_evento)
        fecha_fin_estimacion = retornos.index[indice_fecha_evento - self.gap_estimacion_evento - 1]
        fecha_inicio_estimacion = retornos.index[indice_fecha_evento - self.gap_estimacion_evento - self.L1]
        fecha_inicio_evento = retornos.index[indice_fecha_evento - self.dias_previos]
        fecha_fin_evento = retornos.index[indice_fecha_evento + self.dias_posteriores]

        resultados = {
            'theta_hat': {},
            'car': {},
            'scar': {},
            'var_car': {}
        }

        AR_star_matriz = np.zeros((len(retornos.columns) - 1, self.L2))
        suma_var_AR_star = np.zeros((self.L2, self.L2))

        for ticker in retornos.columns:
            if ticker == self.market_benchmark:
                continue

            est_window = retornos.loc[fecha_inicio_estimacion:fecha_fin_estimacion, [self.market_benchmark, ticker]]
            evt_window = retornos.loc[fecha_inicio_evento:fecha_fin_evento, [self.market_benchmark, ticker]]

            market_model = MarketModel(est_window, evt_window, self.market_benchmark, ticker)
            theta_hat, est_window, evt_window = market_model.calculate_theta_and_ar()

            statistics_calculator = StatisticsCalculator(est_window, evt_window, theta_hat, self.gamma)
            var_AR_star = statistics_calculator.calculate_varianza_ar()
            CAR, SCAR, var_CAR = statistics_calculator.calculate_car_and_scar(var_AR_star)

            resultados['theta_hat'][ticker] = theta_hat
            resultados['car'][ticker] = CAR
            resultados['scar'][ticker] = SCAR
            resultados['var_car'][ticker] = var_CAR

            AR_star_matriz[i, :] = evt_window['AR'].values
            suma_var_AR_star += var_AR_star

        # Aquí continuarías con el cálculo de J1, J2, tests no paramétricos, etc.
        # Y finalmente generarías los gráficos usando la clase Plotter.

# Ejemplo de uso
event_study = EventStudy(
    file_path="precios.csv",
    market_benchmark="^GSPC",
    fecha_evento=pd.to_datetime("2020-06-08"),
    gap_estimacion_evento=10,
    L1=100,
    dias_previos=4,
    dias_posteriores=5
)
event_study.run()
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.gridspec as gridspec
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.")

# -----------------------------
# Data Preparation
# -----------------------------
data = pd.read_csv('all_stocks_5yr.csv')
ibm_data = data[data['Name'] == 'IBM'].copy()
ibm_data['date'] = pd.to_datetime(ibm_data['date'])
ibm_data.sort_values('date', inplace=True)

# Pre-calculate the 20-day moving averages.
window = 20
ibm_data['open_roll'] = ibm_data['open'].rolling(window=window).mean()
ibm_data['high_roll'] = ibm_data['high'].rolling(window=window).mean()
ibm_data['low_roll'] = ibm_data['low'].rolling(window=window).mean()
ibm_data['close_roll'] = ibm_data['close'].rolling(window=window).mean()

# -----------------------------
# Figure Layout Setup
# -----------------------------
fig = plt.figure(figsize=(14, 8))

# Define the main area for full plots (left side).
main_area = [0.05, 0.1, 0.6, 0.85]

# Define preview area as a 3x2 grid on the right.
preview_raw_ax   = fig.add_axes([0.7, 0.6333, 0.125, 0.2667])
preview_ma_ax    = fig.add_axes([0.825, 0.6333, 0.125, 0.2667])
preview_acf_ax   = fig.add_axes([0.7, 0.3667, 0.125, 0.2667])
preview_ar_ax    = fig.add_axes([0.825, 0.3667, 0.125, 0.2667])
preview_arima_ax = fig.add_axes([0.7, 0.1, 0.125, 0.2667])
# (The cell at Row3, Col2 is unused.)

main_axes = []  # To track main area axes

# -----------------------------
# Preview Drawing Functions
# -----------------------------
def draw_raw_preview(ax):
    ax.clear()
    subset = ibm_data.tail(50)
    ax.plot(subset['date'], subset['open'], color='blue')
    ax.plot(subset['date'], subset['high'], color='green')
    ax.plot(subset['date'], subset['low'], color='red')
    ax.plot(subset['date'], subset['close'], color='purple')
    ax.set_title("Raw", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def draw_ma_preview(ax):
    ax.clear()
    subset = ibm_data.tail(50)
    ax.plot(subset['date'], subset['open'], color='blue', alpha=0.5)
    ax.plot(subset['date'], subset['high'], color='green', alpha=0.5)
    ax.plot(subset['date'], subset['low'], color='red', alpha=0.5)
    ax.plot(subset['date'], subset['close'], color='purple', alpha=0.5)
    ax.plot(subset['date'], subset['open'].rolling(window=20).mean(), linestyle='--', color='blue')
    ax.plot(subset['date'], subset['high'].rolling(window=20).mean(), linestyle='--', color='green')
    ax.plot(subset['date'], subset['low'].rolling(window=20).mean(), linestyle='--', color='red')
    ax.plot(subset['date'], subset['close'].rolling(window=20).mean(), linestyle='--', color='purple')
    ax.set_title("MA", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def draw_acf_preview(ax):
    ax.clear()
    plot_acf(ibm_data['open'], ax=ax, lags=20, zero=False)
    ax.set_title("ACF/PACF", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_ar_preview(ax):
    ax.clear()
    ax.text(0.5, 0.5, "AR Model", horizontalalignment='center',
            verticalalignment='center', fontsize=10, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def draw_arima_preview(ax):
    ax.clear()
    ax.text(0.5, 0.5, "ARIMA Model", horizontalalignment='center',
            verticalalignment='center', fontsize=10, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

# Draw preview thumbnails.
draw_raw_preview(preview_raw_ax)
draw_ma_preview(preview_ma_ax)
draw_acf_preview(preview_acf_ax)
draw_ar_preview(preview_ar_ax)
draw_arima_preview(preview_arima_ax)

preview_mapping = {
    preview_raw_ax: 'raw',
    preview_ma_ax: 'ma',
    preview_acf_ax: 'acf',
    preview_ar_ax: 'ar',
    preview_arima_ax: 'arima'
}

# -----------------------------
# Main Area Update Functions
# -----------------------------
def clear_main_area():
    global main_axes
    for ax in main_axes:
        try:
            ax.remove()
        except Exception:
            pass
    main_axes = []

def draw_raw_main():
    clear_main_area()
    ax = fig.add_axes(main_area)
    ax.plot(ibm_data['date'], ibm_data['open'], label='Open', color='blue')
    ax.plot(ibm_data['date'], ibm_data['high'], label='High', color='green')
    ax.plot(ibm_data['date'], ibm_data['low'], label='Low', color='red')
    ax.plot(ibm_data['date'], ibm_data['close'], label='Close', color='purple')
    ax.set_title("IBM Stock Prices - Raw Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    main_axes.append(ax)

def draw_ma_main():
    clear_main_area()
    ax = fig.add_axes(main_area)
    ax.plot(ibm_data['date'], ibm_data['open'], label='Open', color='blue', alpha=0.5)
    ax.plot(ibm_data['date'], ibm_data['high'], label='High', color='green', alpha=0.5)
    ax.plot(ibm_data['date'], ibm_data['low'], label='Low', color='red', alpha=0.5)
    ax.plot(ibm_data['date'], ibm_data['close'], label='Close', color='purple', alpha=0.5)
    ax.plot(ibm_data['date'], ibm_data['open'].rolling(window=20).mean(), linestyle='--', color='blue')
    ax.plot(ibm_data['date'], ibm_data['high'].rolling(window=20).mean(), linestyle='--', color='green')
    ax.plot(ibm_data['date'], ibm_data['low'].rolling(window=20).mean(), linestyle='--', color='red')
    ax.plot(ibm_data['date'], ibm_data['close'].rolling(window=20).mean(), linestyle='--', color='purple')
    ax.set_title("IBM Stock Prices with 20-Day Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    main_axes.append(ax)

def draw_acf_main():
    clear_main_area()
    left = main_area[0]
    bottom = main_area[1]
    right = main_area[0] + main_area[2]
    top = main_area[1] + main_area[3]
    gs = gridspec.GridSpec(4, 2, left=left, bottom=bottom, right=right, top=top, hspace=0.5)
    series_list = [('Open', 'open'), ('High', 'high'), ('Low', 'low'), ('Close', 'close')]
    for i, (label, col) in enumerate(series_list):
        ax1 = fig.add_subplot(gs[i, 0])
        plot_acf(ibm_data[col], ax=ax1, lags=40, zero=False)
        ax1.set_title(f'{label} ACF', fontsize=9)
        main_axes.append(ax1)
        ax2 = fig.add_subplot(gs[i, 1])
        plot_pacf(ibm_data[col], ax=ax2, lags=40, zero=False)
        ax2.set_title(f'{label} PACF', fontsize=9)
        main_axes.append(ax2)

def draw_ar_main():
    clear_main_area()
    ax = fig.add_axes(main_area)
    series = ibm_data.set_index('date')['close']
    series = series.asfreq('B').ffill()
    
    model = AutoReg(series, lags=1)
    model_fit = model.fit()
    forecast = model_fit.predict(start=len(series), end=len(series) + 4)
    forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
    forecast.index = forecast_dates

    print("AR(1) Forecasted prices:")
    print(forecast)
    
    ax.plot(series.index, series, label='Close Price', color='purple')
    ax.plot(forecast.index, forecast, marker='o', linestyle='--', color='orange', label='AR Forecast')
    ax.set_title("AR(1) Model Forecast for IBM Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    
    coef = model_fit.params.iloc[1]
    interpretation_text = (f"AR(1) coefficient: {coef:.3f}\n"
                           "Interpretation: Tomorrow's price is roughly "
                           f"{coef:.3f} times today's price plus noise.")
    ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    main_axes.append(ax)

def draw_arima_main():
    clear_main_area()
    ax = fig.add_axes(main_area)
    series = ibm_data.set_index('date')['close']
    series = series.asfreq('B').ffill()
    
    # Grid search over some ARIMA(p,d,q) combinations.
    p_vals = [0, 1, 2]
    d_vals = [0, 1]
    q_vals = [0, 1, 2]
    
    best_aic = float('inf')
    best_order = None
    best_model_fit = None
    
    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                try:
                    model = ARIMA(series, order=(p,d,q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p,d,q)
                        best_model_fit = model_fit
                except Exception:
                    continue
                    
    # Forecast the next 5 business days.
    forecast = best_model_fit.forecast(steps=5)
    forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
    forecast.index = forecast_dates
    
    print("ARIMA Forecasted prices:")
    print(forecast)
    
    ax.plot(series.index, series, label='Close Price', color='purple')
    ax.plot(forecast.index, forecast, marker='o', linestyle='--', color='orange', label='ARIMA Forecast')
    ax.set_title(f"ARIMA Model Forecast for IBM Close Price\nBest order: {best_order}, AIC: {best_aic:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    # Ensure that the x-axis covers both historical and forecast data.
    ax.set_xlim(series.index[0], forecast.index[-1])
    
    info_text = f"Best ARIMA order: {best_order}\nAIC: {best_aic:.2f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    main_axes.append(ax)
    plt.draw()

def update_main_view(view):
    if view == 'raw':
        draw_raw_main()
    elif view == 'ma':
        draw_ma_main()
    elif view == 'acf':
        draw_acf_main()
    elif view == 'ar':
        draw_ar_main()
    elif view == 'arima':
        draw_arima_main()
    fig.canvas.draw_idle()
    plt.pause(0.001)  # force the GUI to update

# -----------------------------
# Click Event Handler
# -----------------------------
def on_click(event):
    if event.inaxes in preview_mapping:
        view = preview_mapping[event.inaxes]
        update_main_view(view)

fig.canvas.mpl_connect('button_press_event', on_click)

# Initialize with the "raw" view.
update_main_view('raw')
plt.show()

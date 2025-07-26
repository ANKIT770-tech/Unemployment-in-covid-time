import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- 1. Data Acquisition / Simulation ---
try:
    df = pd.read_csv('C:\\Users\\ANKIT\\OneDrive\\Desktop\\INFOBYTE\\intrn\\data\\covid\\Unemployment in India')
    # # Example of parsing date if it's not automatically recognized
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print("Placeholder for loading real data from a CSV.")
    print("Uncomment and modify the 'pd.read_csv' line when you have your data.")

    # For demonstration, we'll proceed with simulated data if no real CSV is loaded.
    data_loaded_from_csv = False

except FileNotFoundError:
    print("CSV file not found. Proceeding with simulated data for demonstration.")
    data_loaded_from_csv = False
except Exception as e:
    print(f"Error loading CSV: {e}. Proceeding with simulated data for demonstration.")
    data_loaded_from_csv = False

# --- Option B: Simulate Data (for demonstration purposes if no real data) ---
if not data_loaded_from_csv:
    # Generate a date range from January 2018 to December 2024
    dates = pd.date_range(start='2018-01-01', end='2024-12-31', freq='MS') # MS = Month Start

    # Simulate a baseline unemployment rate (e.g., 5-8%)
    np.random.seed(42) # for reproducibility
    base_unemployment = np.random.uniform(5.0, 8.0, len(dates))

    # Introduce a 'COVID-19' spike
    covid_start = pd.to_datetime('2020-03-01')
    covid_peak = pd.to_datetime('2020-05-01')
    covid_end = pd.to_datetime('2021-06-01') # Gradual recovery

    spike_factor = []
    for date in dates:
        if covid_start <= date <= covid_peak:
            # Linear increase to peak
            months_into_covid = (date - covid_start).days / 30.0
            factor = 1 + (months_into_covid / 2) * 1.5 # Increase by up to 150%
            spike_factor.append(factor)
        elif covid_peak < date < covid_end:
            # Gradual decrease from peak
            months_from_peak = (date - covid_peak).days / 30.0
            initial_spike = 1 + (2 / 2) * 1.5 # Peak factor from above
            factor = initial_spike - (months_from_peak / 14) * (initial_spike - 1) # Recover over ~14 months
            spike_factor.append(max(1.0, factor)) # Ensure it doesn't go below 1
        else:
            spike_factor.append(1.0) # No impact

    spike_factor = np.array(spike_factor)
    unemployment_rate = base_unemployment * spike_factor

    # Add some noise
    unemployment_rate += np.random.normal(0, 0.5, len(dates))
    unemployment_rate = np.clip(unemployment_rate, 2.0, 25.0) # Ensure realistic range

    # Simulate demographic data (Urban/Rural, Male/Female)
    # Assume Urban was hit harder initially, Rural had a slower recovery in some cases
    unemployment_urban = unemployment_rate * (1 + np.sin(np.arange(len(dates))/12 * 2*np.pi/3) * 0.05 + np.random.normal(0, 0.2, len(dates)))
    unemployment_rural = unemployment_rate * (1 - np.sin(np.arange(len(dates))/12 * 2*np.pi/3) * 0.03 + np.random.normal(0, 0.2, len(dates)))
    unemployment_urban = np.clip(unemployment_urban * 1.1, 2.0, 30.0) # Urban slightly higher, more volatile
    unemployment_rural = np.clip(unemployment_rural * 0.9, 2.0, 20.0)

    unemployment_male = unemployment_rate * (1 - np.cos(np.arange(len(dates))/12 * 2*np.pi/2) * 0.02 + np.random.normal(0, 0.1, len(dates)))
    unemployment_female = unemployment_rate * (1 + np.cos(np.arange(len(dates))/12 * 2*np.pi/2) * 0.03 + np.random.normal(0, 0.15, len(dates)))
    unemployment_male = np.clip(unemployment_male, 2.0, 25.0)
    unemployment_female = np.clip(unemployment_female * 1.1, 2.0, 30.0) # Female unemployment often higher, especially in informal sectors

    df = pd.DataFrame({
        'Date': dates,
        'Overall_Unemployment_Rate': unemployment_rate,
        'Urban_Unemployment_Rate': unemployment_urban,
        'Rural_Unemployment_Rate': unemployment_rural,
        'Male_Unemployment_Rate': unemployment_male,
        'Female_Unemployment_Rate': unemployment_female,
    })
    df.set_index('Date', inplace=True)

print("Data preparation complete.")
print("Using simulated data for demonstration purposes." if not data_loaded_from_csv else "Using loaded CSV data.")


# --- 2. Basic Data Exploration ---
print("\n--- Data Head ---")
print(df.head())

print("\n--- Data Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

# Ensure all columns are numeric for plotting
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors will turn non-numeric into NaN

# --- 3. Time Series Visualization (Overall Unemployment) ---
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Overall_Unemployment_Rate'], label='Overall Unemployment Rate', color='blue')
plt.title('Unemployment Rate Trend Over Time (Simulated Data)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- 4. COVID-19 Impact Visualization ---
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Overall_Unemployment_Rate'], label='Overall Unemployment Rate', color='blue', alpha=0.8)

# Highlight the COVID-19 period (adjust these dates based on your data's actual spike)
covid_start_viz = pd.to_datetime('2020-03-01')
covid_end_viz = pd.to_datetime('2021-06-01') # Period of significant impact/recovery

plt.axvspan(covid_start_viz, covid_end_viz, color='red', alpha=0.2, label='COVID-19 Impact Period')
plt.annotate('COVID-19 Impact',
             xy=(covid_start_viz + (covid_end_viz - covid_start_viz)/2, df['Overall_Unemployment_Rate'].max() * 0.9),
             xytext=(covid_start_viz + (covid_end_viz - covid_start_viz)/2, df['Overall_Unemployment_Rate'].max() * 0.95),
             ha='center', va='bottom',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=10, color='black')

plt.title('Unemployment Rate Highlighting COVID-19 Period')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# --- 5. Comparative Analysis (Urban vs. Rural, Male vs. Female) ---
# Create subplots for better comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Urban vs. Rural
axes[0].plot(df.index, df['Urban_Unemployment_Rate'], label='Urban Unemployment Rate', color='orange')
axes[0].plot(df.index, df['Rural_Unemployment_Rate'], label='Rural Unemployment Rate', color='green')
axes[0].set_title('Unemployment Rate: Urban vs. Rural')
axes[0].set_ylabel('Unemployment Rate (%)')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()
axes[0].axvspan(covid_start_viz, covid_end_viz, color='red', alpha=0.1) # Highlight COVID period

# Male vs. Female
axes[1].plot(df.index, df['Male_Unemployment_Rate'], label='Male Unemployment Rate', color='purple')
axes[1].plot(df.index, df['Female_Unemployment_Rate'], label='Female Unemployment Rate', color='brown')
axes[1].set_title('Unemployment Rate: Male vs. Female')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Unemployment Rate (%)')
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()
axes[1].axvspan(covid_start_viz, covid_end_viz, color='red', alpha=0.1) # Highlight COVID period

plt.tight_layout()
plt.show()

# --- 6. Descriptive Statistics for Different Periods ---
# Define periods
pre_covid_end = '2020-02-29'
during_covid_start = '2020-03-01'
during_covid_end = '2021-06-30' # Adjust based on when recovery was significant
post_covid_start = '2021-07-01'

print("\n--- Unemployment Statistics by Period ---")

# Pre-COVID
pre_covid_df = df.loc[df.index <= pre_covid_end]
print("\nPre-COVID Period (before Mar 2020):")
print(pre_covid_df.describe().loc[['mean', 'std', 'min', 'max']])

# During COVID Peak Impact
during_covid_df = df.loc[(df.index >= during_covid_start) & (df.index <= during_covid_end)]
print(f"\nDuring COVID Impact Period ({during_covid_start} - {during_covid_end}):")
print(during_covid_df.describe().loc[['mean', 'std', 'min', 'max']])

# Post-COVID Recovery/New Normal
post_covid_df = df.loc[df.index >= post_covid_start]
print(f"\nPost-COVID Period (from {post_covid_start}):")
print(post_covid_df.describe().loc[['mean', 'std', 'min', 'max']])


# --- Advanced (Optional) ---
# If you want to dive deeper into time series forecasting, you'd add models like:
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.arima.model import ARIMA
# from prophet import Prophet # Requires 'pip install prophet'

# Example of Decomposition (can be applied to overall or specific categories)
# print("\n--- Time Series Decomposition (Overall Unemployment) ---")
# # Assuming you have at least a year of data and a clear seasonality
# if len(df) > 24: # Need enough data points for monthly seasonality
#     decomposition = seasonal_decompose(df['Overall_Unemployment_Rate'], model='additive', period=12) # period=12 for monthly data
#     fig_decompose = decomposition.plot()
#     fig_decompose.set_size_inches(12, 8)
#     plt.suptitle('Seasonal Decomposition of Overall Unemployment Rate', y=1.02)
#     plt.tight_layout(rect=[0, 0, 1, 0.98])
#     plt.show()
# else:
#     print("Not enough data points for meaningful seasonal decomposition.")

print("\n--- Analysis Complete ---")
print("Remember to replace simulated data with your actual dataset for real insights!")

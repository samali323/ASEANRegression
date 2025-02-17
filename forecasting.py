import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import os


class ASEANForecaster:
    def __init__(self, filepath):
        """
        Initialize the forecaster with data from the given filepath
        """
        # Define indicators first
        self.indicators = [
            'GDP (current US$)',
            'GNI per capita',
            'Industry (% of GDP)',
            'Manufacturing (% of GDP)',
            'Foreign direct investment (% of GDP)',
            'Population, total',
            'Urban population (% of total)',
            'Population growth (annual %)',
            'Energy use (kg of oil equivalent per capita)',
            'Electric power consumption (kWh per capita)',
            'Access to electricity (% of population)',
            'Fossil fuel energy consumption (% of total)',
            'Individuals using the Internet (% of population)',
            'Mobile cellular subscriptions (per 100 people)',
            'PM2.5 air pollution (micrograms per cubic meter)',
            'CO2 emissions (metric tons per capita)',
            'Total greenhouse gas emissions (per capita)',
            'Demand (TWh)'
        ]
        # Load and preprocess data
        self.data = self.load_and_preprocess_data(filepath)

        # Ensure forecasts directory exists
        os.makedirs('forecasts', exist_ok=True)

    def clean_numeric_column(self, series):
        """
        Clean numeric columns by removing currency symbols and commas
        """
        series = series.astype(str)
        cleaned = (series.str.replace('$', '')   # Remove dollar sign
                   .str.replace(',', '')         # Remove commas
                   .str.strip())                 # Remove whitespace
        cleaned = cleaned.replace(['', 'nan', 'NaN'], np.nan)
        return pd.to_numeric(cleaned, errors='coerce')

    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the data
        """
        # Read the CSV
        df = pd.read_csv(filepath)
        # Convert Year to datetime
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        # Clean numeric columns
        numeric_columns = self.indicators.copy()
        numeric_columns.remove('Demand (TWh)')  # Handle separately if needed
        for col in numeric_columns:
            df[col] = self.clean_numeric_column(df[col])
        return df

    def prepare_multivariate_forecast(self, indicator):
        """
        Prepare multivariate forecast for a specific indicator
        Considers relationships between indicators
        """
        # Prepare data for all countries
        countries = self.data['Country'].unique()
        country_forecasts = {}
        for country in countries:
            # Filter data for the specific country
            country_data = self.data[self.data['Country'] == country].set_index('Year')
            # Select historical data up to 2022
            historical_data = country_data.loc[country_data.index.year <= 2022, self.indicators]
            # Interpolate missing values
            historical_data = historical_data.interpolate(method='linear')
            # Remove any remaining NaN rows
            historical_data = historical_data.dropna()
            # If insufficient data, skip
            if len(historical_data) < 5:
                print(f"Insufficient data for {country} - {indicator}")
                continue
            try:
                # Standardize data
                scaler = StandardScaler()
                scaled_data = pd.DataFrame(
                    scaler.fit_transform(historical_data),
                    columns=historical_data.columns,
                    index=historical_data.index
                )
                # Fit VAR model
                var_model = VAR(scaled_data)
                var_results = var_model.fit(maxlags=2)
                # Forecast all indicators
                forecast_scaled = var_results.forecast(scaled_data.values, steps=13)
                # Inverse transform the forecast
                forecast = pd.DataFrame(
                    scaler.inverse_transform(forecast_scaled),
                    columns=historical_data.columns
                )
                # Set index to future years
                last_year = historical_data.index[-1].year
                forecast.index = pd.date_range(
                    start=str(last_year + 1),
                    periods=13,
                    freq='YE'
                )
                # Add country column
                forecast['Country'] = country
                # Store forecast
                country_forecasts[country] = forecast
                # Visualize
                self.visualize_forecast(
                    historical_data[indicator],
                    forecast[indicator],
                    country,
                    indicator
                )
            except Exception as e:
                print(f"Error forecasting {indicator} for {country}: {e}")
        # Combine forecasts
        combined_forecast = pd.concat(list(country_forecasts.values()))
        # Save combined forecast
        combined_forecast.to_csv(f'forecasts/{indicator}_forecast.csv')
        return combined_forecast

    def visualize_forecast(self, historical_data, forecast, country, indicator):
        """
        Visualize forecast for a specific country and indicator
        """
        plt.figure(figsize=(10, 6))
        # Historical data
        plt.plot(historical_data.index, historical_data, label='Historical', color='blue')
        # Forecast
        plt.plot(forecast.index, forecast, label='Forecast', color='red')
        # Confidence intervals
        std_error = np.std(forecast)
        plt.fill_between(
            forecast.index,
            forecast - 1.96 * std_error,
            forecast + 1.96 * std_error,
            color='pink',
            alpha=0.3
        )
        plt.title(f'{country} - {indicator} Forecast')
        plt.xlabel('Year')
        plt.ylabel(indicator)
        plt.legend()
        plt.tight_layout()
        # Save plot
        plt.savefig(f'forecasts/{country}_{indicator}_forecast.png')
        plt.close()

    def forecast_all_indicators(self):
        """
        Forecast all indicators
        """
        all_indicator_forecasts = {}
        for indicator in self.indicators:
            print(f"Forecasting {indicator}")
            forecast = self.prepare_multivariate_forecast(indicator)
            all_indicator_forecasts[indicator] = forecast
        return all_indicator_forecasts


def main():
    # Initialize forecaster
    forecaster = ASEANForecaster('ASEAN_2035.csv')
    # Forecast all indicators
    forecasts = forecaster.forecast_all_indicators()
    print("Forecasting complete!")


if __name__ == "__main__":
    main()

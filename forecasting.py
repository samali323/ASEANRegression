import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import self
import seaborn as sns
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
        cleaned = (series.str.replace('$', '')  # Remove dollar sign
                   .str.replace(',', '')  # Remove commas
                   .str.strip())  # Remove whitespace
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
        Considers relationships between indicators and applies constraints
        """
        # Define indicator-specific constraints
        constraints = {
            'GDP (current US$)': {'min_factor': 0.9},  # No more than 10% below historical minimum
            'GNI per capita': {'min_factor': 0.9},
            'Industry (% of GDP)': {'min_value': 0, 'max_value': 100},  # Percentage constraint
            'Manufacturing (% of GDP)': {'min_value': 0, 'max_value': 100},
            'Foreign direct investment (% of GDP)': {'min_value': -100, 'max_value': 100},  # Can be negative
            'Population, total': {'min_factor': 0.95},  # Population cannot drop sharply
            'Urban population (% of total)': {'min_value': 0, 'max_value': 100},
            'Population growth (annual %)': {'min_value': -10, 'max_value': 10},  # Realistic bounds
            'Energy use (kg of oil equivalent per capita)': {'min_factor': 0.8},
            'Electric power consumption (kWh per capita)': {'min_factor': 0.8},
            'Access to electricity (% of population)': {'min_value': 0, 'max_value': 100, 'min_factor': 1.0},
            # Should not decline
            'Fossil fuel energy consumption (% of total)': {'min_value': 0, 'max_value': 100},
            'Individuals using the Internet (% of population)': {'min_value': 0, 'max_value': 100},
            'Mobile cellular subscriptions (per 100 people)': {'min_value': 0, 'max_value': 200},  # Can exceed 100
            'PM2.5 air pollution (micrograms per cubic meter)': {'min_value': 0},
            'CO2 emissions (metric tons per capita)': {'min_factor': 0.5},  # Allow significant reduction
            'Total greenhouse gas emissions (per capita)': {'min_factor': 0.5},
            'Demand (TWh)': {'min_factor': 0.8}  # Energy demand cannot drop sharply
        }

        # Prepare data for all countries
        countries = self.data['Country'].unique()
        country_forecasts = {}
        for country in countries:
            try:
                # Filter data for the specific country
                country_data = self.data[self.data['Country'] == country].set_index('Year')
                # Select historical data up to 2023
                historical_data = country_data.loc[country_data.index.year <= 2023, self.indicators]
                historical_data = historical_data.interpolate(method='linear')
                # Remove any remaining NaN rows
                historical_data = historical_data.dropna()
                # If insufficient data, skip
                if len(historical_data) < 5:
                    print(f"Insufficient data for {country} - {indicator}. Skipping...")
                    continue
                # Check for constant values
                if np.all(historical_data.values == historical_data.values[0]):
                    print(f"All values are constant for {country} - {indicator}. Skipping...")
                    continue
                # Standardize data
                scaler = StandardScaler()
                scaled_data = pd.DataFrame(
                    scaler.fit_transform(historical_data),
                    columns=historical_data.columns,
                    index=historical_data.index
                )
                # Set frequency explicitly
                scaled_data = scaled_data.asfreq('YS')
                # Fit VAR model with dynamic maxlags
                maxlags = min(len(scaled_data) // 2, 2)
                var_model = VAR(scaled_data)
                var_results = var_model.fit(maxlags=maxlags)
                # Forecast all indicators (adjust steps to 12)
                forecast_scaled = var_results.forecast(scaled_data.values, steps=12)
                forecast = pd.DataFrame(
                    scaler.inverse_transform(forecast_scaled),
                    columns=historical_data.columns
                )
                # Define last year and set forecast index
                last_year = historical_data.index[-1].year
                forecast.index = pd.date_range(start=f"{last_year + 1}", periods=12, freq='YS')  # Match steps=12
                forecast['Country'] = country

                # Apply constraints based on indicator-specific logic
                if indicator in constraints:
                    constraint = constraints[indicator]
                    min_value = constraint.get('min_value', None)
                    max_value = constraint.get('max_value', None)
                    min_factor = constraint.get('min_factor', None)

                    # Apply minimum value constraint
                    if min_value is not None:
                        forecast[indicator] = forecast[indicator].clip(lower=min_value)
                    # Apply maximum value constraint
                    if max_value is not None:
                        forecast[indicator] = forecast[indicator].clip(upper=max_value)
                    # Apply minimum factor constraint
                    if min_factor is not None:
                        min_historical_value = historical_data[indicator].min()
                        lower_bound = min_factor * min_historical_value
                        forecast[indicator] = forecast[indicator].clip(lower=lower_bound)

                # Save forecast for this country and indicator
                self.save_forecast_by_country_and_indicator(country, indicator, forecast)
                # Plot actuals and forecasts
                self.plot_actuals_and_forecast(country, indicator, historical_data[indicator], forecast[indicator])
                # Store forecast for combination
                country_forecasts[country] = forecast
            except Exception as e:
                print(f"Error forecasting {indicator} for {country}: {e}")

        # Handle case where no valid forecasts are generated
        if not country_forecasts:
            print(f"No valid forecasts generated for {indicator}. Skipping combination.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Combine forecasts for all countries
        combined_forecast = pd.concat(list(country_forecasts.values()))
        return combined_forecast

    def save_forecast_by_country_and_indicator(self, country, indicator, forecast):
        """
        Save forecast data to a CSV file
        """
        # Create directory for forecasts if it doesn't exist
        forecasts_dir = os.path.join("forecasts", country)
        os.makedirs(forecasts_dir, exist_ok=True)  # Ensure the directory exists

        # Define file path for the forecast
        forecast_path = os.path.join(forecasts_dir, f"{indicator}.csv")

        # Save forecast to CSV
        forecast.to_csv(forecast_path, index_label="Year")
        print(f"Saved forecast for {country} - {indicator} to {forecast_path}")

    def plot_actuals_and_forecast(self, country, indicator, actuals, forecast):
        """
        Plot actual data (up to 2023) and forecasted data (2024–2035)
        """
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join("plots", country)
        os.makedirs(plots_dir, exist_ok=True)  # Ensure the directory exists

        # Define file path for the plot
        plot_path = os.path.join(plots_dir, f"{indicator}.png")

        # Plot actuals and forecasts
        plt.figure(figsize=(10, 6))
        plt.plot(actuals, label="Actual Data (up to 2023)", color="blue", marker="o")
        plt.plot(forecast, label="Forecast (2024–2035)", color="orange", linestyle="--", marker="x")

        # Add labels and title
        plt.title(f"{country} - {indicator}\nActual vs Forecast", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(indicator, fontsize=12)
        plt.axvline(x=pd.to_datetime("2023-12-31"), color="red", linestyle="--", linewidth=1.5,
                    label="Forecast Start (2024)")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for {country} - {indicator} to {plot_path}")

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

    def plot_correlation_matrix(self):
        """
        Plot a correlation matrix for the indicators
        """
        print("Generating correlation matrix plot...")

        if not hasattr(self, 'data') or self.data is None:
            raise AttributeError("Data has not been initialized. Ensure load_and_preprocess_data has been called.")

        # Combine data for all countries
        combined_data = self.data.groupby('Year')[self.indicators].mean().dropna()

        # Compute correlation matrix
        correlation_matrix = combined_data.corr()

        num_indicators = len(self.indicators)
        plt.figure(
            figsize=(num_indicators * 1.2, num_indicators * 1.2))  # Scale figure size with the number of indicators
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True
        )
        plt.title("Correlation Matrix of Indicators", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        # Save the plot with a higher DPI
        os.makedirs("plots", exist_ok=True)
        plot_path = os.path.join("correlation_matrix.png")
        plt.savefig(plot_path, dpi=400, bbox_inches='tight')  # Set DPI to 300
        plt.close()

        print(f"Saved correlation matrix plot to {plot_path}")


def main():
    # Initialize forecaster
    forecaster = ASEANForecaster('ASEAN_2035.csv')

    # Plot correlation matrix
    forecaster.plot_correlation_matrix()

    # Forecast all indicators
    forecasts = forecaster.forecast_all_indicators()
    print("Forecasting complete!")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import self
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class ASEANProphetForecaster:
    def __init__(self, filepath):
        """
        Initialize the forecaster with data from the given filepath
        """
        # Define indicators
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

        # Create directories for outputs
        os.makedirs(os.path.join('forecasts', 'Prophet', 'forecasts'), exist_ok=True)
        os.makedirs(os.path.join('plots', 'Prophet', 'plots'), exist_ok=True)

    def clean_numeric_column(self, series):
        """Clean numeric columns by removing currency symbols and commas"""
        series = series.astype(str)
        cleaned = (series.str.replace('$', '')
                   .str.replace(',', '')
                   .str.strip())
        cleaned = cleaned.replace(['', 'nan', 'NaN'], np.nan)
        return pd.to_numeric(cleaned, errors='coerce')

    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the data"""
        df = pd.read_csv(filepath)
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')

        # Clean numeric columns
        numeric_columns = self.indicators.copy()
        numeric_columns.remove('Demand (TWh)')  # Handle separately if needed
        for col in numeric_columns:
            df[col] = self.clean_numeric_column(df[col])
        return df

    def apply_constraints(self, forecast, historical_data, indicator):
        """Apply constraints to forecasted values"""
        constraints = {
            'GDP (current US$)': {'min_factor': 0.9},
            'GNI per capita': {'min_factor': 0.9},
            'Industry (% of GDP)': {'min_value': 0, 'max_value': 100},
            'Manufacturing (% of GDP)': {'min_value': 0, 'max_value': 100},
            'Foreign direct investment (% of GDP)': {'min_value': -100, 'max_value': 100},
            'Population, total': {'min_factor': 0.95},
            'Urban population (% of total)': {'min_value': 0, 'max_value': 100},
            'Population growth (annual %)': {'min_value': -10, 'max_value': 10},
            'Energy use (kg of oil equivalent per capita)': {'min_factor': 0.8},
            'Electric power consumption (kWh per capita)': {'min_factor': 0.8},
            'Access to electricity (% of population)': {'min_value': 0, 'max_value': 100, 'min_factor': 1.0},
            'Fossil fuel energy consumption (% of total)': {'min_value': 0, 'max_value': 100},
            'Individuals using the Internet (% of population)': {'min_value': 0, 'max_value': 100},
            'Mobile cellular subscriptions (per 100 people)': {'min_value': 0, 'max_value': 200},
            'PM2.5 air pollution (micrograms per cubic meter)': {'min_value': 0},
            'CO2 emissions (metric tons per capita)': {'min_factor': 0.5},
            'Total greenhouse gas emissions (per capita)': {'min_factor': 0.5},
            'Demand (TWh)': {'min_factor': 0.8}
        }

        if indicator in constraints:
            constraint = constraints[indicator]

            # Apply min/max value constraints
            if 'min_value' in constraint:
                forecast['yhat'] = forecast['yhat'].clip(lower=constraint['min_value'])
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=constraint['min_value'])
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=constraint['min_value'])

            if 'max_value' in constraint:
                forecast['yhat'] = forecast['yhat'].clip(upper=constraint['max_value'])
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(upper=constraint['max_value'])
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=constraint['max_value'])

            # Apply minimum factor constraints
            if 'min_factor' in constraint:
                min_historical = historical_data.min()
                lower_bound = min_historical * constraint['min_factor']
                forecast['yhat'] = forecast['yhat'].clip(lower=lower_bound)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=lower_bound)
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=lower_bound)

        return forecast

    def get_model_config(self, indicator, historical_data):
        """Get Prophet model configuration based on indicator type"""
        base_config = {
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False
        }

        # Economic indicators
        if indicator in ['GDP (current US$)', 'GNI per capita']:
            return {
                **base_config,
                'growth': 'linear',
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10,
                'changepoint_range': 0.9
            }

        # Population metrics
        elif indicator in ['Population, total', 'Urban population (% of total)', 'Population growth (annual %)']:
            return {
                **base_config,
                'growth': 'logistic' if 'total' in indicator else 'linear',
                'changepoint_prior_scale': 0.001,
                'seasonality_prior_scale': 0.1,
                'changepoint_range': 0.98
            }

        # Energy metrics
        elif any(term in indicator for term in ['Energy', 'electricity', 'power', 'TWh']):
            return {
                **base_config,
                'growth': 'linear',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 1.0,
                'changepoint_range': 0.95
            }

        # Environmental metrics
        elif any(term in indicator for term in ['CO2', 'greenhouse', 'pollution']):
            return {
                **base_config,
                'growth': 'linear',
                'changepoint_prior_scale': 0.15,
                'seasonality_prior_scale': 5.0,
                'changepoint_range': 0.9
            }

        # Technology adoption metrics
        elif any(term in indicator for term in ['Internet', 'Mobile', 'subscriptions']):
            return {
                **base_config,
                'growth': 'logistic',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 0.1,
                'changepoint_range': 0.95
            }

        # Percentage-based metrics
        elif '(%)' in indicator or '(% of GDP)' in indicator:
            return {
                **base_config,
                'growth': 'linear',
                'changepoint_prior_scale': 0.03,
                'seasonality_prior_scale': 1.0,
                'changepoint_range': 0.95
            }

        # Default configuration
        return {
            **base_config,
            'growth': 'linear',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 1.0,
            'changepoint_range': 0.95
        }

    def save_forecast(self, country, indicator, forecast):
        """Save forecast data to CSV"""
        # Create directory for forecasts
        forecast_dir = os.path.join('forecasts', 'Prophet', 'forecasts', country)
        os.makedirs(forecast_dir, exist_ok=True)

        # Prepare results DataFrame
        results = pd.DataFrame({
            'Year': forecast['ds'],
            'Forecast': forecast['yhat'],
            'Lower_Bound': forecast['yhat_lower'],
            'Upper_Bound': forecast['yhat_upper']
        })

        # Save to CSV
        filepath = os.path.join(forecast_dir, f"{indicator}.csv")
        results.to_csv(filepath, index=False)
        print(f"    Saved forecast to {filepath}")

    def plot_forecast(self, country, indicator, historical_data, forecast):
        """Plot actual data and forecast"""
        # Create directory for plots
        plots_dir = os.path.join('plots', 'Prophet', 'plots', country)
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, f"{indicator}.png")

        plt.figure(figsize=(10, 6))

        # Plot historical data
        plt.plot(historical_data.index, historical_data.values,
                 label="Actual Data (up to 2023)", color="blue", marker="o")

        # Plot forecast
        plt.plot(forecast['ds'], forecast['yhat'],
                 label="Forecast (2024–2035)", color="orange", linestyle="--", marker="x")

        # Add confidence interval
        plt.fill_between(forecast['ds'],
                         forecast['yhat_lower'],
                         forecast['yhat_upper'],
                         color='orange', alpha=0.2,
                         label='95% Confidence Interval')

        # Add labels and title
        plt.title(f"{country} - {indicator}\nActual vs Forecast", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(indicator, fontsize=12)
        plt.axvline(x=pd.to_datetime("2023-12-31"), color="red", linestyle="--", linewidth=1.5,
                    label="Forecast Start (2024)")
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved plot to {plot_path}")

    def plot_correlation_matrix(self):
        """Plot correlation matrix for indicators"""
        print("Generating correlation matrix plot...")

        if not hasattr(self, 'data') or self.data is None:
            raise AttributeError("No data available for correlation matrix")

        # Combine data for all countries
        combined_data = self.data.groupby('Year')[self.indicators].mean().dropna()

        # Compute correlation matrix
        correlation_matrix = combined_data.corr()

        # Create plot
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True
        )

        plt.title("Correlation Matrix of Indicators", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Save plot
        plots_dir = os.path.join('plots', 'Prophet')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, 'correlation_matrix.png')
        plt.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"    Saved correlation matrix to {plot_path}")

    def forecast_indicator(self, country, indicator):
        """Forecast a specific indicator for a country using Prophet"""
        print(f"\n  Data diagnostics for {country} - {indicator}:")
        try:
            # Filter data for the specific country
            country_data = self.data[self.data['Country'] == country].set_index('Year')

            # Select historical data up to 2023
            historical_data = country_data.loc[country_data.index.year <= 2023, indicator]

            # Print initial data diagnostics
            print(f"    Initial data shape: {historical_data.shape}")
            print(f"    Non-null count: {historical_data.count()}")

            # Check if we have enough valid data
            valid_count = historical_data.count()
            if valid_count == 0:
                print(f"    ERROR: No valid data points found for {indicator}")
                return None

            print(f"    Value range: [{historical_data.min()}, {historical_data.max()}]")

            # Handle missing values with improved strategy
            if historical_data.isna().any():
                missing_count = historical_data.isna().sum()
                missing_years = historical_data.index[historical_data.isna()].tolist()
                print(f"    Missing values: {missing_count}")
                print(f"    Missing years: {missing_years}")

                # Special handling for recent missing values
                recent_missing = [yr for yr in missing_years if yr.year > 2020]
                if recent_missing:
                    print(f"    Found {len(recent_missing)} recent missing values (after 2020)")
                    # Use last available value for recent missing data
                    last_value = historical_data.dropna().iloc[-1]
                    for year in recent_missing:
                        historical_data[year] = last_value
                    print(f"    Filled recent missing values with last available value: {last_value}")

                # Enhanced interpolation strategy
                temp_data = historical_data.copy()

                # First try cubic interpolation
                temp_data = temp_data.interpolate(method='cubic')

                # Then fill any remaining gaps with linear interpolation
                temp_data = temp_data.interpolate(method='linear', limit_direction='both')

                # Finally, if we still have gaps at the start/end, use nearest
                temp_data = temp_data.interpolate(method='nearest', limit_direction='both')

                if temp_data.isna().sum() > 0:
                    print(f"    ERROR: Unable to interpolate all missing values")
                    return None

                historical_data = temp_data
                print(f"    Successfully interpolated missing values")
                print(f"    New value range: [{historical_data.min()}, {historical_data.max()}]")

            # Check for constant values with improved handling
            if len(set(historical_data.dropna())) == 1:
                value = historical_data.iloc[0]
                print(f"    WARNING: All values are constant ({value})")
                if value != 0:  # Only add variation if value is non-zero
                    variation = value * 0.001
                    historical_data = historical_data + np.random.normal(0, variation, len(historical_data))
                    print(f"    Added small random variation (std: {variation:.6f})")
                else:
                    print(f"    Keeping constant zero values")

            # Print final data stats
            print(f"    Final data stats:")
            print(f"      Time range: {historical_data.index.min()} to {historical_data.index.max()}")
            print(f"      Number of unique values: {len(historical_data.unique())}")
            print(f"      Mean: {historical_data.mean():.2f}")
            print(f"      Std: {historical_data.std():.2f}")

            # Prepare Prophet data
            prophet_data = pd.DataFrame({
                'ds': historical_data.index,
                'y': historical_data.values
            })

            # Configure Prophet model
            model_config = {
                'yearly_seasonality': True,
                'growth': 'linear',
                'changepoint_prior_scale': 0.05
            }

            # Check if this is a percentage-based indicator
            is_percentage = (
                    indicator.endswith('(%)') or
                    indicator.endswith('(% of GDP)') or
                    'percent' in indicator.lower() or
                    'rate' in indicator.lower()
            )

            if is_percentage:
                model_config['growth'] = 'logistic'
                prophet_data['floor'] = 0
                prophet_data['cap'] = 100

            # Initialize and fit Prophet model
            m = Prophet(**model_config)
            m.fit(prophet_data)

            # Create future dataframe using YE instead of Y
            future = pd.DataFrame({'ds': pd.date_range(
                start='2024',
                end='2035',
                freq='YE'  # Use YE instead of Y
            )})

            if is_percentage:
                future['floor'] = 0
                future['cap'] = 100

            # Make forecast
            forecast = m.predict(future)

            # Apply constraints
            forecast = self.apply_constraints(forecast, historical_data, indicator)

            # Save forecast
            self.save_forecast(country, indicator, forecast)

            # Plot results
            self.plot_forecast(country, indicator, historical_data, forecast)

            return forecast

        except Exception as e:
            print(f"Error forecasting {indicator} for {country}: {e}")
            return None
    def forecast_all_indicators(self):
        """Forecast all indicators for all countries"""
        for country in self.data['Country'].unique():
            print(f"\nProcessing forecasts for {country}")
            for indicator in self.indicators:
                print(f"  Forecasting {indicator}")
                self.forecast_indicator(country, indicator)

    def save_correlation_matrix(self):
        """Save correlation matrix plot"""
        os.makedirs(os.path.join('plots', 'Prophet'), exist_ok=True)
        plot_path = os.path.join('plots', 'Prophet', 'correlation_matrix.png')
        plt.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close()

        print("Saved correlation matrix plot")


def main():
    # Initialize forecaster
    forecaster = ASEANProphetForecaster('ASEAN_2035.csv')

    # Plot correlation matrix
    forecaster.plot_correlation_matrix()

    # Forecast all indicators
    forecaster.forecast_all_indicators()

    print("\nForecasting complete!")


if __name__ == "__main__":
    main()

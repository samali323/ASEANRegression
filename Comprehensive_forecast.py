import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from prophet import Prophet
import os
import argparse
from datetime import datetime

class ASEANForecaster:
    def __init__(self, filepath, country=None):
        """
        Initialize the forecaster with data from the given filepath

        Parameters:
        filepath (str): Path to the CSV data file
        country (str, optional): If provided, only analyze data for this country
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
            'Demand (TWh)',
            # Add the 12 new Era Mean variables
            'JanEraMean',
            'FebEraMean',
            'MarEraMean',
            'AprEraMean',
            'MayEraMean',
            'JunEraMean',
            'JulEraMean',
            'AugEraMean',
            'SepEraMean',
            'OctEraMean',
            'NovEraMean',
            'DecEraMean'
        ]

        # Load and preprocess data
        self.data = self.load_and_preprocess_data(filepath)

        # Filter by country if specified
        self.selected_country = country
        if self.selected_country:
            if self.selected_country not in self.data['Country'].unique():
                available_countries = ", ".join(self.data['Country'].unique())
                raise ValueError(f"Country '{self.selected_country}' not found in dataset. Available countries: {available_countries}")

            print(f"Analyzing data for {self.selected_country} only")
            self.data = self.data[self.data['Country'] == self.selected_country]

        # Create base directory structure
        self.create_directory_structure()

    def create_directory_structure(self):
        """Create the directory structure for outputs"""
        countries = self.data['Country'].unique()

        for country in countries:
            # Convert country to string to ensure it's a valid directory name
            country_str = str(country)

            # Create VAR directories
            os.makedirs(os.path.join('output', country_str, 'VAR', 'forecasts'), exist_ok=True)
            os.makedirs(os.path.join('output', country_str, 'VAR', 'plots'), exist_ok=True)

            # Create Prophet directories
            os.makedirs(os.path.join('output', country_str, 'Prophet', 'forecasts'), exist_ok=True)
            os.makedirs(os.path.join('output', country_str, 'Prophet', 'plots'), exist_ok=True)

        print(f"Created directory structure for {len(countries)} countries")

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

        # Ensure Country column is treated as string
        if 'Country' in df.columns:
            df['Country'] = df['Country'].astype(str)

        # Convert Year to datetime
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')

        # Clean numeric columns
        known_indicators = [ind for ind in self.indicators if ind in df.columns]
        for col in known_indicators:
            df[col] = self.clean_numeric_column(df[col])

        # Generate mock data for the 12 Era Mean variables if they don't exist
        era_means = [col for col in self.indicators if col.endswith('EraMean')]
        for col in era_means:
            if col not in df.columns:
                print(f"Generating mock data for {col}")
                # Create random values based on the 'Demand (TWh)' column if it exists
                if 'Demand (TWh)' in df.columns:
                    base_values = df['Demand (TWh)'].fillna(df['Demand (TWh)'].mean())
                    # Create slightly different monthly values (±10%)
                    df[col] = base_values * np.random.uniform(0.9, 1.1, size=len(df))
                else:
                    # Default to random values between 10 and 100
                    df[col] = np.random.uniform(10, 100, size=len(df))

        return df

    def plot_correlation_matrix(self):
        """
        Plot a correlation matrix for the indicators
        """
        print("Generating correlation matrix plot...")

        if not hasattr(self, 'data') or self.data is None:
            raise AttributeError("Data has not been initialized.")

        # Combine data for all countries
        combined_data = self.data.groupby('Year')[self.indicators].mean().dropna()

        # Compute correlation matrix
        correlation_matrix = combined_data.corr()

        plt.figure(figsize=(15, 15))
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

        # Save the plot
        if self.selected_country:
            plot_path = os.path.join('output', self.selected_country, 'correlation_matrix.png')
        else:
            plot_path = os.path.join('output', 'correlation_matrix.png')

        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"Saved correlation matrix plot to {plot_path}")

    def var_forecast(self, country, indicator):
        """
        Perform VAR forecasting for a specific country and indicator
        """
        # Convert country to string to ensure consistency
        country = str(country)

        print(f"VAR Forecasting {indicator} for {country}")

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
                print(f"  Insufficient data for {country} - {indicator}. Skipping...")
                return None

            # Check for constant values
            if np.all(historical_data[indicator].values == historical_data[indicator].values[0]):
                print(f"  All values are constant for {country} - {indicator}. Skipping...")
                return None

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
            forecast.index = pd.date_range(start=f"{last_year + 1}", periods=12, freq='YS')
            forecast['Country'] = country

            # Apply constraints based on indicator-specific logic
            forecast = self.apply_constraints(indicator, forecast, historical_data[indicator])

            # Save forecast for this country and indicator
            self.save_var_forecast(country, indicator, forecast)

            # Plot actuals and forecasts
            self.plot_var_forecast(country, indicator, historical_data[indicator], forecast[indicator])

            return forecast

        except Exception as e:
            print(f"  Error in VAR forecasting for {indicator} - {country}: {e}")
            return None

    def save_var_forecast(self, country, indicator, forecast):
        """
        Save VAR forecast data to a CSV file
        """
        # Define file path for the forecast
        forecast_dir = os.path.join('output', country, 'VAR', 'forecasts')
        os.makedirs(forecast_dir, exist_ok=True)

        forecast_path = os.path.join(forecast_dir, f"{indicator}.csv")

        # Save forecast to CSV
        forecast.to_csv(forecast_path, index_label="Year")
        print(f"  Saved VAR forecast for {country} - {indicator} to {forecast_path}")

    def plot_var_forecast(self, country, indicator, actuals, forecast):
        """
        Plot actual data and VAR forecasted data
        """
        # Define file path for the plot
        plots_dir = os.path.join('output', country, 'VAR', 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, f"{indicator}.png")

        # Plot actuals and forecasts
        plt.figure(figsize=(10, 6))
        plt.plot(actuals, label="Actual Data (up to 2023)", color="blue", marker="o")
        plt.plot(forecast, label="VAR Forecast (2024–2035)", color="orange", linestyle="--", marker="x")

        # Add labels and title
        plt.title(f"{country} - {indicator}\nActual vs VAR Forecast", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(indicator, fontsize=12)
        plt.axvline(x=pd.to_datetime("2023-12-31"), color="red", linestyle="--", linewidth=1.5,
                    label="Forecast Start (2024)")
        plt.legend()
        plt.grid(True)

        # Set appropriate y-axis limits based on indicator type
        self.set_appropriate_y_limits(plt, indicator, actuals, forecast)

        # Save the plot
        plt.savefig(plot_path)
        plt.close()
        print(f"  Saved VAR plot for {country} - {indicator} to {plot_path}")

    def prophet_forecast(self, country, indicator):
        """
        Perform Prophet forecasting for a specific country and indicator
        """
        # Convert country to string to ensure consistency
        country = str(country)

        print(f"Prophet Forecasting {indicator} for {country}")

        try:
            # Filter data for the specific country
            country_data = self.data[self.data['Country'] == country].set_index('Year')

            # Select historical data up to 2023
            historical_data = country_data.loc[country_data.index.year <= 2023, indicator]

            # Print initial data diagnostics
            print(f"  Initial data shape: {historical_data.shape}")
            print(f"  Non-null count: {historical_data.count()}")

            # Check if we have enough valid data
            valid_count = historical_data.count()
            if valid_count < 5:
                print(f"  ERROR: Not enough valid data points found for {indicator}")
                return None

            # Handle missing values with improved strategy
            if historical_data.isna().any():
                missing_count = historical_data.isna().sum()
                print(f"  Missing values: {missing_count}")

                # Enhanced interpolation strategy
                historical_data = historical_data.interpolate(method='cubic')
                historical_data = historical_data.interpolate(method='linear', limit_direction='both')
                historical_data = historical_data.interpolate(method='nearest', limit_direction='both')

                if historical_data.isna().sum() > 0:
                    print(f"  ERROR: Unable to interpolate all missing values")
                    return None

                print(f"  Successfully interpolated missing values")

            # Check for constant values
            if len(set(historical_data.dropna())) == 1:
                value = historical_data.iloc[0]
                print(f"  WARNING: All values are constant ({value})")
                if value != 0:  # Only add variation if value is non-zero
                    variation = value * 0.001
                    historical_data = historical_data + np.random.normal(0, variation, len(historical_data))
                    print(f"  Added small random variation (std: {variation:.6f})")

            # Prepare Prophet data
            prophet_data = pd.DataFrame({
                'ds': historical_data.index,
                'y': historical_data.values
            })

            # Configure Prophet model
            model_config = self.get_prophet_config(indicator, historical_data)

            # Initialize and fit Prophet model
            m = Prophet(**model_config)
            m.fit(prophet_data)

            # Create future dataframe
            future = pd.DataFrame({'ds': pd.date_range(
                start='2024',
                end='2035',
                freq='YE'
            )})

            # Add floor/cap if using logistic growth
            if model_config.get('growth') == 'logistic':
                if indicator.endswith('(%)') or indicator.endswith('(% of GDP)'):
                    future['floor'] = 0
                    future['cap'] = 100

            # Make forecast
            forecast = m.predict(future)

            # Apply constraints
            forecast = self.apply_prophet_constraints(indicator, forecast, historical_data)

            # Save forecast
            self.save_prophet_forecast(country, indicator, forecast)

            # Plot results
            self.plot_prophet_forecast(country, indicator, historical_data, forecast)

            return forecast

        except Exception as e:
            print(f"  Error in Prophet forecasting for {indicator} - {country}: {e}")
            return None

    def get_prophet_config(self, indicator, historical_data):
        """
        Get Prophet model configuration based on indicator type
        """
        base_config = {
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False
        }

        # Monthly indicators (Era Mean variables)
        if 'EraMean' in indicator:
            return {
                **base_config,
                'growth': 'linear',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'changepoint_range': 0.9,
                'seasonality_mode': 'multiplicative'  # Good for seasonal data
            }

        # Economic indicators
        elif indicator in ['GDP (current US$)', 'GNI per capita']:
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

    def save_prophet_forecast(self, country, indicator, forecast):
        """
        Save Prophet forecast data to CSV
        """
        # Create directory for forecasts
        forecast_dir = os.path.join('output', country, 'Prophet', 'forecasts')
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
        print(f"  Saved Prophet forecast to {filepath}")

    def plot_prophet_forecast(self, country, indicator, historical_data, forecast):
        """
        Plot actual data and Prophet forecast
        """
        # Create directory for plots
        plots_dir = os.path.join('output', country, 'Prophet', 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, f"{indicator}.png")

        plt.figure(figsize=(10, 6))

        # Plot historical data
        plt.plot(historical_data.index, historical_data.values,
                 label="Actual Data (up to 2023)", color="blue", marker="o")

        # Plot forecast
        plt.plot(forecast['ds'], forecast['yhat'],
                 label="Prophet Forecast (2024–2035)", color="orange", linestyle="--", marker="x")

        # Add confidence interval
        plt.fill_between(forecast['ds'],
                         forecast['yhat_lower'],
                         forecast['yhat_upper'],
                         color='orange', alpha=0.2,
                         label='95% Confidence Interval')

        # Add labels and title
        plt.title(f"{country} - {indicator}\nActual vs Prophet Forecast", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(indicator, fontsize=12)
        plt.axvline(x=pd.to_datetime("2023-12-31"), color="red", linestyle="--", linewidth=1.5,
                    label="Forecast Start (2024)")
        plt.legend()
        plt.grid(True)

        # Set appropriate y-axis limits based on indicator type
        self.set_appropriate_y_limits(plt, indicator, historical_data, forecast['yhat'])

        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved Prophet plot to {plot_path}")

    def set_appropriate_y_limits(self, plt, indicator, historical_data, forecast_data):
        """
        Set appropriate y-axis limits based on indicator type to avoid misleading visualizations
        """
        # Get min and max values from both historical and forecast data
        if isinstance(historical_data, pd.Series):
            hist_min = historical_data.min()
            hist_max = historical_data.max()
        else:
            hist_min = np.min(historical_data)
            hist_max = np.max(historical_data)

        if isinstance(forecast_data, pd.Series):
            fore_min = forecast_data.min()
            fore_max = forecast_data.max()
        else:
            fore_min = np.min(forecast_data)
            fore_max = np.max(forecast_data)

        data_min = min(hist_min, fore_min)
        data_max = max(hist_max, fore_max)

        # Special handling for specific indicators

        # Access to electricity should show full context for 0-100% range
        if indicator == 'Access to electricity (% of population)':
            # If the values are all near 100%, show at least 90-100% to provide context
            if data_min > 90:
                plt.ylim(90, 101)
            else:
                # Otherwise show a reasonable range based on data
                plt.ylim(max(0, data_min - 5), min(101, data_max + 5))

        # For percentage indicators, ensure appropriate context
        elif '(%)' in indicator or indicator.endswith('(% of GDP)') or indicator.endswith('(% of total)'):
            # If values are all in upper range (> 80%)
            if data_min > 80:
                plt.ylim(80, 101)
            # If values are all in middle range (40-80%)
            elif data_min > 40:
                plt.ylim(40, 101)
            # If values are in lower range but not near zero
            elif data_min > 10:
                plt.ylim(max(0, data_min - 10), min(101, data_max + 10))
            # For values near zero, provide appropriate context
            else:
                plt.ylim(0, min(101, data_max + 10))

        # For non-percentage indicators, use a reasonable buffer
        else:
            # Calculate range
            data_range = data_max - data_min

            # If all values are very close, expand the range to show context
            if data_range < 0.01 * data_max:
                # For near-constant data, show reasonable variation
                buffer = 0.1 * data_max
                plt.ylim(max(0, data_min - buffer), data_max + buffer)
            else:
                # Otherwise add a 5% buffer on each side
                buffer = 0.05 * data_range
                plt.ylim(max(0, data_min - buffer), data_max + buffer)

    def apply_constraints(self, indicator, forecast, historical_data):
        """
        Apply constraints to VAR forecasted values
        """
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
            'Access to electricity (% of population)': {'min_value': 0, 'max_value': 100, 'min_factor': 1.0},  # Should not decline
            'Fossil fuel energy consumption (% of total)': {'min_value': 0, 'max_value': 100},
            'Individuals using the Internet (% of population)': {'min_value': 0, 'max_value': 100},
            'Mobile cellular subscriptions (per 100 people)': {'min_value': 0, 'max_value': 200},  # Can exceed 100
            'PM2.5 air pollution (micrograms per cubic meter)': {'min_value': 0},
            'CO2 emissions (metric tons per capita)': {'min_factor': 0.5},  # Allow significant reduction
            'Total greenhouse gas emissions (per capita)': {'min_factor': 0.5},
            'Demand (TWh)': {'min_factor': 0.8}  # Energy demand cannot drop sharply
        }

        # Add constraints for the Era Mean variables
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            constraints[f'{month}EraMean'] = {'min_factor': 0.5, 'min_value': 0}

        # Special handling for Access to electricity for developed countries
        if indicator == 'Access to electricity (% of population)':
            last_value = historical_data.iloc[-1] if isinstance(historical_data, pd.Series) else historical_data[-1]
            if last_value >= 99.5:
                # For countries with near-universal access, maintain at least the last historical value
                forecast[indicator] = forecast[indicator].clip(lower=last_value)

        # Apply constraint if it exists for this indicator
        if indicator in constraints:
            constraint = constraints[indicator]

            # Apply minimum value constraint
            if 'min_value' in constraint:
                forecast[indicator] = forecast[indicator].clip(lower=constraint['min_value'])

            # Apply maximum value constraint
            if 'max_value' in constraint:
                forecast[indicator] = forecast[indicator].clip(upper=constraint['max_value'])

            # Apply minimum factor constraint
            if 'min_factor' in constraint:
                min_historical_value = historical_data.min() if isinstance(historical_data, pd.Series) else np.min(historical_data)
                if min_historical_value > 0:  # Only apply if minimum value is positive
                    lower_bound = constraint['min_factor'] * min_historical_value
                    forecast[indicator] = forecast[indicator].clip(lower=lower_bound)

        return forecast

    def apply_prophet_constraints(self, indicator, forecast, historical_data):
        """
        Apply constraints to Prophet forecasted values
        """
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

        # Add constraints for the Era Mean variables
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            constraints[f'{month}EraMean'] = {'min_factor': 0.5, 'min_value': 0}

        # Special handling for Access to electricity for developed countries
        if indicator == 'Access to electricity (% of population)':
            last_value = historical_data.iloc[-1] if isinstance(historical_data, pd.Series) else historical_data[-1]
            if last_value >= 99.5:
                # For countries with near-universal access, maintain at least the last historical value
                forecast['yhat'] = forecast['yhat'].clip(lower=last_value)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=last_value)
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=last_value)

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
                min_historical = historical_data.min() if isinstance(historical_data, pd.Series) else np.min(historical_data)
                if min_historical > 0:  # Only apply if minimum value is positive
                    lower_bound = min_historical * constraint['min_factor']
                    forecast['yhat'] = forecast['yhat'].clip(lower=lower_bound)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=lower_bound)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=lower_bound)

        return forecast

    def run_var_forecasts(self):
        """
        Run VAR forecasts for all countries and indicators
        """
        if self.selected_country:
            countries = [self.selected_country]
        else:
            countries = self.data['Country'].unique()

        for country in countries:
            # Convert country to string to ensure consistency
            country = str(country)

            print(f"\nRunning VAR forecasts for {country}")
            for indicator in self.indicators:
                if indicator in self.data.columns:
                    self.var_forecast(country, indicator)
                else:
                    print(f"Indicator {indicator} not found in dataset. Skipping...")

    def run_prophet_forecasts(self):
        """
        Run Prophet forecasts for all countries and indicators
        """
        if self.selected_country:
            countries = [self.selected_country]
        else:
            countries = self.data['Country'].unique()

        for country in countries:
            # Convert country to string to ensure consistency
            country = str(country)

            print(f"\nRunning Prophet forecasts for {country}")
            for indicator in self.indicators:
                if indicator in self.data.columns:
                    self.prophet_forecast(country, indicator)
                else:
                    print(f"Indicator {indicator} not found in dataset. Skipping...")

    def combine_forecasts(self):
        """
        Combine VAR and Prophet forecasts into a single dataset for each country

        Optimized to reduce DataFrame fragmentation
        """
        if self.selected_country:
            countries = [self.selected_country]
        else:
            countries = self.data['Country'].unique()

        print("\nCombining forecasts...")

        for country in countries:
            # Convert country to string to ensure consistency
            country = str(country)

            print(f"  Processing {country}...")

            # Create a list of available indicators in the dataset
            country_hist = self.data[self.data['Country'] == country].copy()
            available_indicators = [ind for ind in self.indicators if ind in country_hist.columns]

            # Initialize data dictionaries to store all columns at once
            data_dict = {'Year': range(2000, 2036), 'Country': country}

            # Step 1: First, gather all historical data
            years_range = range(2000, 2036)

            for indicator in available_indicators:
                # Initialize columns with NaN
                data_dict[indicator] = [np.nan] * len(years_range)
                data_dict[f'{indicator}_Source'] = [None] * len(years_range)

                # Get historical values (2000-2023)
                historical_values = country_hist.set_index('Year')[indicator]

                # Add historical data (2000-2023)
                for i, year in enumerate(years_range):
                    if year <= 2023 and year in historical_values.index.year:
                        idx = historical_values.index.year.get_loc(year)
                        data_dict[indicator][i] = historical_values.iloc[idx]
                        data_dict[f'{indicator}_Source'][i] = 'Historical'

            # Step 2: Create the base DataFrame at once
            base_df = pd.DataFrame(data_dict)

            # Step 3: Add forecast data for each indicator
            for indicator in available_indicators:
                # Gather VAR forecasts data
                var_values = {}
                var_file = os.path.join('output', country, 'VAR', 'forecasts', f"{indicator}.csv")

                if os.path.exists(var_file):
                    try:
                        var_df = pd.read_csv(var_file)
                        var_df['Year'] = pd.to_datetime(var_df['Year']).dt.year

                        # Get forecast values from 2024 onwards
                        forecast_data = var_df[var_df['Year'] >= 2024]

                        if indicator in forecast_data.columns:
                            for _, row in forecast_data.iterrows():
                                var_values[row['Year']] = row[indicator]
                            print(f"    Added VAR forecast for {indicator}")
                    except Exception as e:
                        print(f"    Error loading VAR forecast for {indicator}: {e}")

                # Gather Prophet forecasts data
                prophet_values = {'forecast': {}, 'lower': {}, 'upper': {}}
                prophet_file = os.path.join('output', country, 'Prophet', 'forecasts', f"{indicator}.csv")

                if os.path.exists(prophet_file):
                    try:
                        prophet_df = pd.read_csv(prophet_file)
                        prophet_df['Year'] = pd.to_datetime(prophet_df['Year']).dt.year

                        # Get forecast data from 2024 onwards
                        prophet_df = prophet_df[prophet_df['Year'] >= 2024]

                        for _, row in prophet_df.iterrows():
                            year = row['Year']
                            prophet_values['forecast'][year] = row['Forecast']
                            prophet_values['lower'][year] = row['Lower_Bound']
                            prophet_values['upper'][year] = row['Upper_Bound']
                        print(f"    Added Prophet forecast for {indicator}")
                    except Exception as e:
                        print(f"    Error loading Prophet forecast for {indicator}: {e}")

                # Apply forecasts to the dataframe
                if var_values or prophet_values['forecast']:
                    # Create copy of the dataframe to defragment
                    base_df = base_df.copy()

                    # Add VAR forecasts
                    for i, row in base_df.iterrows():
                        year = row['Year']
                        if year >= 2024 and year in var_values:
                            base_df.at[i, indicator] = var_values[year]
                            base_df.at[i, f'{indicator}_Source'] = 'VAR_Forecast'

                    # Add Prophet forecasts (only if data exists)
                    if prophet_values['forecast']:
                        # Create new columns if they don't exist
                        if f'{indicator}_Prophet' not in base_df.columns:
                            base_df[f'{indicator}_Prophet'] = np.nan
                            base_df[f'{indicator}_Lower_CI'] = np.nan
                            base_df[f'{indicator}_Upper_CI'] = np.nan

                        for i, row in base_df.iterrows():
                            year = row['Year']
                            if year >= 2024 and year in prophet_values['forecast']:
                                base_df.at[i, f'{indicator}_Prophet'] = prophet_values['forecast'][year]
                                base_df.at[i, f'{indicator}_Lower_CI'] = prophet_values['lower'][year]
                                base_df.at[i, f'{indicator}_Upper_CI'] = prophet_values['upper'][year]

            # Save combined data
            output_dir = os.path.join('output', country)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{country}_combined_forecast.csv')
            base_df.to_csv(output_file, index=False)
            print(f"  Saved combined data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='ASEAN Forecaster')
    parser.add_argument('--method', type=str, choices=['var', 'prophet', 'both'], default='both',
                        help='Forecasting method to use (default: both)')
    parser.add_argument('--file', type=str, default='ASEAN_2035.csv',
                        help='Input CSV file (default: ASEAN_2035.csv)')

    args = parser.parse_args()

    # Initialize forecaster without selecting a country yet
    temp_forecaster = ASEANForecaster(args.file)

    # Get list of available countries
    available_countries = temp_forecaster.data['Country'].unique()
    available_countries = [str(country) for country in available_countries]

    # Display available countries
    print("\nAvailable countries:")
    for i, country in enumerate(available_countries, 1):
        print(f"{i}. {country}")

    # Ask user to select a country
    while True:
        try:
            selection = input("\nSelect a country (enter number or name) or 'all' for all countries: ")

            # Check if user wants all countries
            if selection.lower() == 'all':
                selected_country = None
                print("Analyzing all countries")
                break

            # Check if selection is a number
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(available_countries):
                    selected_country = available_countries[index]
                    print(f"Selected: {selected_country}")
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(available_countries)}")
            # Check if selection is a country name
            elif selection in available_countries:
                selected_country = selection
                print(f"Selected: {selected_country}")
                break
            else:
                print("Invalid selection. Please try again.")
        except Exception as e:
            print(f"Error: {e}")

    # Re-initialize forecaster with selected country
    print(f"\nStarting ASEAN Forecaster with parameters:")
    print(f"  Country: {selected_country if selected_country else 'All countries'}")
    print(f"  Method: {args.method}")
    print(f"  File: {args.file}")

    forecaster = ASEANForecaster(args.file, selected_country)

    # Plot correlation matrix
    forecaster.plot_correlation_matrix()

    # Run forecasts
    if args.method in ['var', 'both']:
        forecaster.run_var_forecasts()

    if args.method in ['prophet', 'both']:
        forecaster.run_prophet_forecasts()

    # Combine forecasts
    if args.method == 'both':
        forecaster.combine_forecasts()

    print("\nForecasting complete!")

if __name__ == "__main__":
    main()

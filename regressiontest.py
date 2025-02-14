import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob


def prepare_data(file_path, country):
    """Prepare data for forecasting"""
    # Read and filter data
    df = pd.read_csv(file_path)
    df = df[df['Country'] == country].copy()

    # Create base prophet dataframe
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Year'].astype(str), format='%Y'),
        'y': df['Demand (TWh)']
    })

    # List of features to include
    features = [
        'Population', 'GDP(USD)',
        'Industry (% of GDP)', 'Manufacturing (% of GDP)',
        'Foreign direct investment (% of GDP)',
        'Energy use (kg of oil equivalent per capita)',
        'Electric power consumption (kWh per capita)',
        'Access to electricity (% of population)',
        'Electric power transmission and distribution losses (% of output)',
        'Fossil fuel energy consumption (% of total)',
        'Urban population (% of total)',
        'Individuals using the Internet (% of population)',
        'Mobile cellular subscriptions (per 100 people)',
        'PM2.5 air pollution (micrograms per cubic meter)'
    ]

    # Add selected features to prophet dataframe
    for feature in features:
        if feature in df.columns:
            # Handle missing values using interpolation
            series = df[feature].copy()
            series = series.interpolate(method='linear', limit_direction='both')
            series = series.ffill().bfill()  # Handle any remaining NaNs
            prophet_df[feature] = series

            # Standardize the feature
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:  # Only standardize if std > 0
                prophet_df[f'{feature}_std'] = (series - mean_val) / std_val

    return prophet_df


def create_forecast(df, end_year=2035):
    """Create forecast using Prophet with enhanced indicators"""
    # Calculate historical growth rate
    historical_cagr = (df['y'].iloc[-1] / df['y'].iloc[0]) ** (1 / len(df)) - 1
    print(f"Historical CAGR: {historical_cagr * 100:.1f}%")

    # Initialize Prophet with conservative parameters
    model = Prophet(
        growth='linear',
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=0.01,
        seasonality_mode='multiplicative',
        yearly_seasonality=False
    )

    # Add standardized regressors
    std_columns = [col for col in df.columns if col.endswith('_std')]
    for col in std_columns:
        if not df[col].isnull().any():  # Only add if no NaN values
            model.add_regressor(col)

    # Add growth cap
    df['cap'] = df['y'].iloc[-1] * (1 + max(historical_cagr * 1.2, 0.08)) ** np.arange(len(df))

    # Fit model
    model.fit(df)

    # Create future dataframe
    periods = end_year - df['ds'].dt.year.max()
    future = model.make_future_dataframe(periods=periods, freq='YE')

    # Add cap to future
    future['cap'] = df['y'].iloc[-1] * (1 + max(historical_cagr * 1.2, 0.08)) ** np.arange(len(future))

    # Add standardized features to future
    for col in std_columns:
        if not df[col].isnull().any():
            future[col] = df[col].iloc[-1]  # Use last known value

    # Make prediction
    forecast = model.predict(future)

    # Post-process forecast
    min_growth = 0.02  # 2% minimum annual growth
    max_growth = 0.08  # 8% maximum annual growth

    last_actual = df['y'].iloc[-1]
    for i in range(len(df), len(forecast)):
        if i == len(df):
            # First forecast year
            forecast.loc[i, 'yhat'] = max(
                last_actual * (1 + min_growth),
                min(last_actual * (1 + max_growth), forecast.loc[i, 'yhat'])
            )
        else:
            # Subsequent years
            prev_forecast = forecast.loc[i - 1, 'yhat']
            forecast.loc[i, 'yhat'] = max(
                prev_forecast * (1 + min_growth),
                min(prev_forecast * (1 + max_growth), forecast.loc[i, 'yhat'])
            )

        # Adjust confidence intervals
        forecast.loc[i, 'yhat_lower'] = forecast.loc[i, 'yhat'] * 0.95
        forecast.loc[i, 'yhat_upper'] = forecast.loc[i, 'yhat'] * 1.05

    return model, forecast


def print_forecast_results(model, forecast, df, country):
    """Print comprehensive forecast results"""
    print(f"\nForecast Results for {country}")
    print("=" * 50)

    # Historical analysis
    historical_cagr = (df['y'].iloc[-1] / df['y'].iloc[0]) ** (1 / len(df)) - 1

    print("\nHISTORICAL ANALYSIS:")
    print("-" * 20)
    print(f"Initial Demand (1999): {df['y'].iloc[0]:.1f} TWh")
    print(f"Current Demand (2023): {df['y'].iloc[-1]:.1f} TWh")
    print(f"Historical CAGR (1999-2023): {historical_cagr * 100:.1f}%")

    # Analyze features
    original_features = [col for col in df.columns if not col.endswith('_std') and col not in ['ds', 'y', 'cap']]
    used_features = [col[:-4] for col in df.columns if col.endswith('_std')]
    unused_features = [feat for feat in original_features if feat not in used_features]

    # Calculate key metrics
    current_demand = df['y'].iloc[-1]
    final_forecast = forecast['yhat'].iloc[-1]
    total_growth = (final_forecast - current_demand) / current_demand * 100
    forecast_years = len(forecast) - len(df)
    avg_annual_growth = (final_forecast / current_demand) ** (1 / forecast_years) - 1

    print("\nFEATURE ANALYSIS:")
    print("-" * 20)
    print(f"Total Features Available: {len(original_features)}")
    print(f"Features Used in Forecast: {len(used_features)}")

    if unused_features:
        print("\nFEATURES NOT USED IN FORECAST:")
        for feature in unused_features:
            print(f"- {feature}")
        print("\nReason: Likely due to missing or insufficient data")

    print("\nUSED FEATURES:")
    for feature in used_features:
        print(f"- {feature}")

    print("\nFORECAST SUMMARY:")
    print("-" * 20)
    print(f"Projected Demand (2035): {final_forecast:.1f} TWh")
    print(f"Total Growth (2023-2035): {total_growth:.1f}%")
    print(f"Projected CAGR: {avg_annual_growth * 100:.1f}%")

    print("\nDETAILED FORECAST:")
    print("-" * 20)
    print("\nYear      Demand     Change     Growth Rate")
    print("-" * 45)

    prev_value = current_demand
    for _, row in forecast[len(df):].iterrows():
        annual_change = row['yhat'] - prev_value
        growth_rate = (row['yhat'] / prev_value - 1) * 100
        print(f"{row['ds'].year}    {row['yhat']:8.1f}    {annual_change:8.1f}    {growth_rate:8.1f}%")
        prev_value = row['yhat']


def save_plot(model, forecast, df, country, output_dir='plots'):
    """Save forecast plot"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create and format the forecast plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # Plot historical data
    ax.plot(df['ds'], df['y'], 'k.', label='Historical Data')

    # Plot forecast
    forecast_dates = forecast['ds']
    forecast_values = forecast['yhat']
    ax.plot(forecast_dates, forecast_values, 'b-', label='Forecast')

    # Plot confidence interval
    ax.fill_between(forecast_dates,
                    forecast['yhat_lower'],
                    forecast['yhat_upper'],
                    color='b', alpha=0.2,
                    label='95% Confidence Interval')

    # Formatting
    ax.set_title(f'Electricity Demand Forecast for {country} (1999-2035)', pad=20)
    ax.set_xlabel('Year')
    ax.set_ylabel('Demand (TWh)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Save plot
    filename = f"{output_dir}/{country}_forecast_plot_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Forecast plot saved to: {filename}")


def main():
    """Main function to run the forecast"""
    countries = [
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao People\'s Democratic Republic (the)',
        'Malaysia', 'Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Viet Nam'
    ]

    print("Available countries:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    while True:
        try:
            choice = int(input("\nEnter the number of the country to analyze (1-10): "))
            if 1 <= choice <= len(countries):
                selected_country = countries[choice - 1]
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")

    # Find most recent enhanced dataset
    enhanced_files = glob.glob('ASEANIndicators_Enhanced_*.csv')
    if not enhanced_files:
        print("No enhanced dataset found. Please run the data merge script first.")
        return

    latest_file = max(enhanced_files, key=os.path.getctime)

    # Prepare data
    prophet_df = prepare_data(latest_file, selected_country)

    # Create forecast
    print("\nCreating forecast to 2035...")
    model, forecast = create_forecast(prophet_df, end_year=2035)

    # Print results
    print_forecast_results(model, forecast, prophet_df, selected_country)

    # Save plot
    save_plot(model, forecast, prophet_df, selected_country)

    # Display plot
    plt.show()


if __name__ == "__main__":
    main()

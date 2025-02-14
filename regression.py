import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import os


def prepare_data(file_path, country):
    """Prepare data for Prophet forecasting"""
    df = pd.read_csv(file_path)
    df = df[df['Country'] == country].copy()

    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Year'].astype(str), format='%Y'),
        'y': df['Demand (TWh)'],
        'population': df['Population'],
        'gdp': df['GDP(USD)']
    })

    return prophet_df


def calculate_growth_rates(df, covid_adjustment=True):

    # Extended pre-COVID period (2010-2019)
    pre_covid_data = df[(df['ds'].dt.year >= 2010) & (df['ds'].dt.year <= 2019)].copy()

    # Post-COVID recovery period (2021-2023)
    post_covid_data = df[df['ds'].dt.year >= 2021].copy()

    # Calculate pre-COVID growth rates (2010-2019)
    population_growth_pre = (pre_covid_data['population'].iloc[-1] / pre_covid_data['population'].iloc[0]) ** (1/10) - 1
    gdp_growth_pre = (pre_covid_data['gdp'].iloc[-1] / pre_covid_data['gdp'].iloc[0]) ** (1/10) - 1

    if covid_adjustment:
        # Calculate post-COVID growth rates (2021-2023)
        population_growth_post = (post_covid_data['population'].iloc[-1] / post_covid_data['population'].iloc[0]) ** (1/3) - 1
        gdp_growth_post = (post_covid_data['gdp'].iloc[-1] / post_covid_data['gdp'].iloc[0]) ** (1/3) - 1

        # Weighted blend of pre and post COVID rates
        # Give more weight to pre-COVID for population (more stable trend)
        population_growth = (2 * population_growth_pre + population_growth_post) / 3
        # Give more weight to post-COVID for GDP (recent recovery)
        gdp_growth = (gdp_growth_pre + 2 * gdp_growth_post) / 3

        # Print diagnostic information
        print("\nGrowth Rate Calculation Details:")
        print(f"Pre-COVID Period GDP Growth (2010-2019): {gdp_growth_pre*100:.2f}%")
        print(f"Post-COVID Period GDP Growth (2021-2023): {gdp_growth_post*100:.2f}%")
        print(f"Final Blended GDP Growth Rate: {gdp_growth*100:.2f}%")
        print(f"Pre-COVID Period Population Growth (2010-2019): {population_growth_pre*100:.2f}%")
        print(f"Post-COVID Period Population Growth (2021-2023): {population_growth_post*100:.2f}%")
        print(f"Final Blended Population Growth Rate: {population_growth*100:.2f}%")
    else:
        population_growth = population_growth_pre
        gdp_growth = gdp_growth_pre

    # Ensure growth rates are not unreasonably low
    population_growth = max(population_growth, population_growth_pre)
    gdp_growth = max(gdp_growth, (gdp_growth_pre + gdp_growth_post) / 2)

    return population_growth, gdp_growth

def create_forecast(df, end_year=2035):
    """Create Prophet forecast with adjusted growth rates"""
    periods = end_year - df['ds'].dt.year.max()

    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        growth='linear',
        changepoint_prior_scale=0.1
    )

    # Add regressors
    model.add_regressor('population')
    model.add_regressor('gdp')

    # Fit the model
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='YE')
    future['population'] = np.nan
    future['gdp'] = np.nan

    # Fill historical values
    future.loc[:len(df)-1, 'population'] = df['population'].values
    future.loc[:len(df)-1, 'gdp'] = df['gdp'].values

    # Calculate growth rates with detailed COVID adjustment
    population_growth, gdp_growth = calculate_growth_rates(df, covid_adjustment=True)

    # Project future values with compound growth
    last_population = df['population'].iloc[-1]
    last_gdp = df['gdp'].iloc[-1]

    # Project future values
    for i in range(periods):
        idx = len(df) + i
        # Use compound growth formula
        future.loc[idx, 'population'] = last_population * (1 + population_growth) ** (i + 1)
        future.loc[idx, 'gdp'] = last_gdp * (1 + gdp_growth) ** (i + 1)

    # Make predictions
    forecast = model.predict(future)

    return model, forecast
def save_forecast_data(forecast, df, country, output_dir='forecasts'):
    """Save forecast results to CSV"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new DataFrame instead of modifying a slice
    forecast_data = pd.DataFrame({
        'Year': forecast['ds'].dt.year,
        'Forecast (TWh)': forecast['yhat'],
        'Lower 95%': forecast['yhat_lower'],
        'Upper 95%': forecast['yhat_upper']
    })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/{country}_forecast_{timestamp}.csv"
    forecast_data.to_csv(filename, index=False)
    print(f"\nForecast data saved to: {filename}")


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

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Save plot
    filename = f"{output_dir}/{country}_forecast_plot_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Forecast plot saved to: {filename}")


def print_forecast_results(model, forecast, df, country):
    """Print comprehensive forecast results"""
    print(f"\nForecast Results for {country}")
    print("=" * 50)

    # Calculate growth rates first for use in summary
    population_growth, gdp_growth = calculate_growth_rates(df)

    # Summary at the top
    print("\nKEY FINDINGS:")
    print("-" * 20)
    current_demand = df['y'].iloc[-1]
    final_forecast = forecast['yhat'].iloc[-1]
    total_growth = (final_forecast - current_demand) / current_demand * 100
    avg_annual_growth = (final_forecast / current_demand) ** (1/12) - 1  # 12 years from 2023 to 2035

    print(f"Current Demand (2023): {current_demand:.1f} TWh")
    print(f"Projected Demand (2035): {final_forecast:.1f} TWh")
    print(f"Total Expected Growth: {total_growth:.1f}%")
    print(f"Average Annual Growth Rate: {avg_annual_growth*100:.1f}%")

    # Model Accuracy
    historical_predictions = forecast.iloc[:len(df)]
    mape = np.mean(np.abs((df['y'].values - historical_predictions['yhat'].values) / df['y'].values)) * 100
    rmse = np.sqrt(np.mean((df['y'].values - historical_predictions['yhat'].values)**2))

    print("\nMODEL ACCURACY:")
    print("-" * 20)
    print(f"MAPE: {mape:.2f}% (Less than 10% indicates high accuracy)")
    print(f"RMSE: {rmse:.2f} TWh")

    # Print the projected growth rates
    print("\nPROJECTED ANNUAL GROWTH RATES:")
    print("-" * 20)
    print(f"Population: {population_growth*100:.2f}% per year")
    print(f"GDP: {gdp_growth*100:.2f}% per year")

    # Historical Growth Analysis
    historical_pop_growth = (df['population'].iloc[-1] / df['population'].iloc[0]) ** (1/len(df)) - 1
    historical_gdp_growth = (df['gdp'].iloc[-1] / df['gdp'].iloc[0]) ** (1/len(df)) - 1
    print("\nHISTORICAL GROWTH RATES (1999-2023):")
    print("-" * 20)
    print(f"Historical Population Growth: {historical_pop_growth*100:.2f}% per year")
    print(f"Historical GDP Growth: {historical_gdp_growth*100:.2f}% per year")
    # Detailed Forecast
    print("\nDETAILED FORECAST:")
    print("-" * 20)
    future_forecast = forecast[len(df):][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print("Year      Forecast    Lower 95%    Upper 95%")
    print("-" * 45)
    for _, row in future_forecast.iterrows():
        print(f"{row['ds'].year}    {row['yhat']:9.2f}    {row['yhat_lower']:9.2f}    {row['yhat_upper']:9.2f}")

    print("\nINTERPRETATION:")
    print("-" * 20)
    print(f"• The model shows high accuracy with a MAPE of {mape:.2f}%")
    print(f"• Electricity demand is expected to grow by {total_growth:.1f}% between 2023 and 2035")
    print(f"• The forecast accounts for COVID-19 impact on GDP growth patterns")
    print("• The 95% confidence intervals show the range of possible outcomes")
    print(
        f"• By 2035, demand could range from {future_forecast['yhat_lower'].iloc[-1]:.1f} to {future_forecast['yhat_upper'].iloc[-1]:.1f} TWh")


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

    df = prepare_data(r'C:\Users\samal\Documents\ASEANIndicators.csv', selected_country)

    print("\nCreating forecast to 2035...")
    model, forecast = create_forecast(df, end_year=2035)

    print_forecast_results(model, forecast, df, selected_country)

    save_forecast_data(forecast, df, selected_country)
    save_plot(model, forecast, df, selected_country)

    # Display plot
    plt.show()


if __name__ == "__main__":
    main()

import pandas as pd
import os
import numpy as np

def create_clean_merged_data():
    """Create clean merged data with consistent year range"""
    # Create output directory
    output_dir = 'merged_data'
    os.makedirs(output_dir, exist_ok=True)

    # Load historical data
    historical_df = pd.read_csv('ASEAN_2035.csv')

    # Clean numeric columns
    for col in historical_df.columns:
        if col not in ['Year', 'Country']:
            if historical_df[col].dtype == 'object':
                historical_df[col] = historical_df[col].str.replace('$', '').str.replace(',', '')
            historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')

    # Get list of indicators
    indicators = [col for col in historical_df.columns if col not in ['Year', 'Country']]

    # Process each country
    countries = historical_df['Country'].unique()

    for country in countries:
        print(f"\nProcessing {country}...")

        # Create base DataFrame with all years
        base_df = pd.DataFrame(index=range(2000, 2036))
        base_df['Country'] = country

        # Add historical data (2000-2023)
        country_hist = historical_df[historical_df['Country'] == country].copy()
        for indicator in indicators:
            base_df[indicator] = np.nan
            historical_values = country_hist.set_index('Year')[indicator]
            base_df.loc[2000:2023, indicator] = historical_values
            base_df[f'{indicator}_Source'] = 'Historical'

        # Add VAR forecasts (2024-2035)
        var_path = os.path.join('forecasts', country)
        if os.path.exists(var_path):
            # Load single VAR forecast file
            for indicator in indicators:
                var_file = os.path.join(var_path, f"{indicator}.csv")
                if os.path.exists(var_file):
                    try:
                        var_df = pd.read_csv(var_file)
                        var_df['Year'] = pd.to_datetime(var_df['Year']).dt.year

                        # Get forecast values from 2024 onwards
                        forecast_data = var_df[var_df['Year'] >= 2024]

                        # Check if indicator exists in forecast data
                        if indicator in forecast_data.columns:
                            base_df.loc[2024:2035, indicator] = forecast_data.set_index('Year')[indicator]
                            base_df.loc[2024:2035, f'{indicator}_Source'] = 'VAR_Forecast'
                            print(f"  Added VAR forecast for {indicator}")
                    except Exception as e:
                        print(f"  Error loading VAR forecast for {indicator}: {e}")

        # Add Prophet forecasts (2024-2035)
        prophet_path = os.path.join('forecasts', 'Prophet', 'forecasts', country)
        if os.path.exists(prophet_path):
            for indicator in indicators:
                prophet_file = os.path.join(prophet_path, f"{indicator}.csv")
                if os.path.exists(prophet_file):
                    try:
                        prophet_df = pd.read_csv(prophet_file)
                        prophet_df['Year'] = pd.to_datetime(prophet_df['Year']).dt.year

                        # Get forecast data from 2024 onwards
                        prophet_data = prophet_df[prophet_df['Year'] >= 2024].set_index('Year')

                        base_df.loc[2024:2035, f'{indicator}_Prophet'] = prophet_data['Forecast']
                        base_df.loc[2024:2035, f'{indicator}_Lower_CI'] = prophet_data['Lower_Bound']
                        base_df.loc[2024:2035, f'{indicator}_Upper_CI'] = prophet_data['Upper_Bound']
                        print(f"  Added Prophet forecast for {indicator}")
                    except Exception as e:
                        print(f"  Error loading Prophet forecast for {indicator}: {e}")

        # Reset index to make Year a column
        base_df.index.name = 'Year'
        base_df = base_df.reset_index()

        # Save combined data
        output_file = os.path.join(output_dir, f'{country}_clean_series.csv')
        base_df.to_csv(output_file, index=False)
        print(f"Saved clean data series to {output_file}")

        # Print data verification
        print(f"\nVerification for {country}:")
        print("Sample of historical data (2023):")
        hist_sample = base_df[base_df['Year'] == 2023][indicators].iloc[0]
        print(hist_sample.to_string())

        print("\nSample of VAR forecasts (2024):")
        var_sample = base_df[base_df['Year'] == 2024][indicators].iloc[0]
        print(var_sample.to_string())

        if any(col.endswith('_Prophet') for col in base_df.columns):
            print("\nSample of Prophet forecasts (2024):")
            prophet_cols = [col for col in base_df.columns if col.endswith('_Prophet')]
            prophet_sample = base_df[base_df['Year'] == 2024][prophet_cols].iloc[0]
            print(prophet_sample.to_string())

def main():
    print("Starting clean data merge process...")
    create_clean_merged_data()
    print("\nData merge complete!")

if __name__ == "__main__":
    main()

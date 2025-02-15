import pandas as pd
import numpy as np
from prophet import Prophet
import glob
import os
from datetime import datetime


def identify_percentage_columns(df):
    """Identify columns that represent percentages based on column names and values"""
    percentage_columns = []

    for col in df.columns:
        if col in ['Country', 'Year']:
            continue

        # Check column name for percentage indicators
        if any(term in col.lower() for term in ['%', 'percent', 'rate']):
            percentage_columns.append(col)
            continue

        # Convert to numeric, handling any non-numeric values
        numeric_series = pd.to_numeric(df[col], errors='coerce')

        if numeric_series.notna().any():
            # Check if maximum value is around 100 and minimum is around 0
            max_val = numeric_series.max()
            min_val = numeric_series.min()
            if not pd.isna(max_val) and not pd.isna(min_val):
                if 0 <= min_val <= 105 and 0 <= max_val <= 105:
                    percentage_columns.append(col)

    return percentage_columns


def interpolate_with_prophet(data, is_percentage=False):
    """
    Interpolate missing values using Prophet

    Parameters:
    -----------
    data : pandas.Series
        Time series data with missing values
    is_percentage : bool
        Whether the data represents percentages

    Returns:
    --------
    pandas.Series
        Interpolated data
    """
    # Convert to numeric if not already
    data = pd.to_numeric(data, errors='coerce')

    if len(data.dropna()) < 2:
        return data

    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': pd.to_datetime(data.index.astype(str), format='%Y'),
        'y': data.values
    })

    # Remove any remaining NaN values
    df = df.dropna()

    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        growth='linear',
        changepoint_prior_scale=0.05
    )

    if is_percentage:
        # Add bounds for percentage data
        model.add_regressor('cap', standardize=False)
        model.add_regressor('floor', standardize=False)
        df['cap'] = 100
        df['floor'] = 0

    try:
        model.fit(df)
    except Exception as e:
        print(f"  Warning: Prophet fitting failed: {str(e)}")
        return data

    # Create future dataframe including historical dates
    future = model.make_future_dataframe(periods=0, freq='Y')
    if is_percentage:
        future['cap'] = 100
        future['floor'] = 0

    # Make predictions
    forecast = model.predict(future)

    # Return interpolated values
    interpolated = pd.Series(forecast['yhat'].values, index=data.index)

    if is_percentage:
        # Ensure values stay within 0-100 range
        interpolated = interpolated.clip(0, 100)

    # Only fill missing values
    filled = data.copy()
    filled[data.isna()] = interpolated[data.isna()]

    return filled


def process_country_data(df, country):
    """Process data for a single country"""
    # Convert numeric columns to float
    for col in df.columns:
        if col not in ['Country', 'Year']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get percentage columns
    percentage_columns = identify_percentage_columns(df)
    print(f"\nIdentified percentage columns for {country}:")
    for col in percentage_columns:
        print(f"  - {col}")

    # Process each column
    for column in df.columns:
        if column not in ['Country', 'Year']:
            print(f"\nProcessing {column} for {country}")

            # Get data for this column
            data = df.set_index('Year')[column]

            # Check if column is percentage
            is_percentage = column in percentage_columns

            # Interpolate missing values
            if data.isna().any():
                print(f"  Found {data.isna().sum()} missing values")
                filled_data = interpolate_with_prophet(data, is_percentage)
                df.loc[df['Country'] == country, column] = filled_data.values
                print(f"  Interpolated missing values")
            else:
                print("  No missing values")

    return df


def main():
    # Find most recent combined file
    combined_files = glob.glob('worldbank_data/ASEAN_combined_*.csv')
    if not combined_files:
        print("No combined data file found!")
        return

    latest_file = max(combined_files, key=os.path.getctime)
    print(f"Using file: {latest_file}")

    # Read data
    df = pd.read_csv(latest_file)

    # Process each country
    processed_data = pd.DataFrame()
    for country in df['Country'].unique():
        print(f"\nProcessing data for {country}")
        country_data = df[df['Country'] == country].copy()
        processed_country_data = process_country_data(country_data, country)
        processed_data = pd.concat([processed_data, processed_country_data])

    # Save processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'processed_data/ASEAN_processed_{timestamp}.csv'
    os.makedirs('processed_data', exist_ok=True)
    processed_data.to_csv(output_file, index=False)
    print(f"\nSaved processed data to: {output_file}")

    # Print summary statistics
    print("\nProcessing Summary:")
    print("=" * 50)
    for country in processed_data['Country'].unique():
        country_data = processed_data[processed_data['Country'] == country]
        missing_before = df[df['Country'] == country].isna().sum().sum()
        missing_after = country_data.isna().sum().sum()
        print(f"\n{country}:")
        print(f"  Missing values before: {missing_before}")
        print(f"  Missing values after: {missing_after}")
        print(f"  Filled values: {missing_before - missing_after}")


if __name__ == "__main__":
    main()
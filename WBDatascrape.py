import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os


def get_asean_countries():
    """Return dictionary of ASEAN countries and their World Bank codes"""
    return {
        'Brunei Darussalam': 'BRN',
        'Cambodia': 'KHM',
        'Indonesia': 'IDN',
        'Lao PDR': 'LAO',
        'Malaysia': 'MYS',
        'Myanmar': 'MMR',
        'Philippines': 'PHL',
        'Singapore': 'SGP',
        'Thailand': 'THA',
        'Vietnam': 'VNM'
    }


def get_indicators():
    """Return dictionary of indicators to collect with their World Bank codes"""
    return {
        # Existing Economic Indicators
        'NY.GDP.MKTP.CD': 'GDP (current US$)',
        'NV.IND.TOTL.ZS': 'Industry (% of GDP)',
        'NV.IND.MANF.ZS': 'Manufacturing (% of GDP)',
        'BX.KLT.DINV.WD.GD.ZS': 'Foreign direct investment (% of GDP)',
        'EG.USE.PCAP.KG.OE': 'Energy use (kg of oil equivalent per capita)',

        # New Indicators
        'EG.CFT.ACCS.ZS': 'Access to clean fuels and technologies for cooking',
        'EG.RNW.TOTL.ZS': 'Renewable energy consumption (% of total final energy consumption)',
        'EG.IMP.CONS.ZS': 'Energy imports (% of energy use)',
        'SP.POP.GROW': 'Population growth rate',
        'NY.GNP.PCAP.CD': 'GNI per capita',
        'EN.ATM.GHGT.KT.CE': 'Total greenhouse gas emissions',

        # Existing Energy Indicators
        'EG.USE.ELEC.KH.PC': 'Electric power consumption (kWh per capita)',
        'EG.ELC.ACCS.ZS': 'Access to electricity (% of population)',
        'EG.ELC.LOSS.ZS': 'Electric power transmission and distribution losses (% of output)',
        'EG.USE.COMM.FO.ZS': 'Fossil fuel energy consumption (% of total)',

        # Existing Development Indicators
        'SP.URB.TOTL.IN.ZS': 'Urban population (% of total)',
        'SP.POP.TOTL': 'Population, total',
        'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)',
        'IT.CEL.SETS.P2': 'Mobile cellular subscriptions (per 100 people)',

        # Existing Environmental Indicators
        'EN.ATM.CO2E.PC': 'CO2 emissions (metric tons per capita)',
        'EN.ATM.PM25.MC.M3': 'PM2.5 air pollution (micrograms per cubic meter)'
    }
def fetch_wb_data(country_code, indicator, start_year, end_year):
    """
    Fetch data from World Bank API using requests
    """
    base_url = "http://api.worldbank.org/v2/country"
    url = f"{base_url}/{country_code}/indicator/{indicator}?format=json&date={start_year}:{end_year}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:  # World Bank API returns metadata in first element
                values = {}
                for entry in data[1]:
                    if entry['value'] is not None:
                        values[int(entry['date'])] = entry['value']
                return pd.Series(values)
        return pd.Series()
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.Series()


def collect_worldbank_data(start_year=2000, end_year=2022):
    """
    Collect World Bank data for ASEAN countries
    """
    countries = get_asean_countries()
    indicators = get_indicators()

    # Create output directory
    output_dir = 'worldbank_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize dictionary for country data
    country_data = {}

    print(f"Collecting World Bank data for {start_year}-{end_year}")
    print(f"Number of indicators: {len(indicators)}\n")

    for country_name, country_code in countries.items():
        print(f"Collecting data for {country_name}...")

        # Initialize DataFrame for this country
        country_df = pd.DataFrame(index=range(start_year, end_year + 1))
        country_df.index.name = 'Year'

        # Collect each indicator
        for indicator_code, indicator_name in indicators.items():
            print(f"Collecting {indicator_name}...", end=' ')

            series = fetch_wb_data(country_code, indicator_code, start_year, end_year)

            if not series.empty:
                country_df[indicator_name] = series
                print(f"✓ ({len(series)} years of data)")
            else:
                print("✗ No data available")

        # Save country data
        country_data[country_name] = country_df
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/{country_name.replace(' ', '_')}_{timestamp}.csv"
        country_df.to_csv(filename)

        # Print data availability summary
        print(f"\nData availability for {country_name}:")
        for column in country_df.columns:
            available = country_df[column].count()
            total = len(country_df)
            print(f"{column}: {available}/{total} years ({available / total * 100:.1f}%)")
        print(f"\nSaved to: {filename}\n")

    # Create and save combined dataset
    print("Creating combined dataset...")
    combined_data = pd.DataFrame()

    for country_name, df in country_data.items():
        df_copy = df.copy()
        df_copy['Country'] = country_name
        combined_data = pd.concat([combined_data, df_copy])

    combined_filename = f"{output_dir}/ASEAN_combined_{timestamp}.csv"
    combined_data.to_csv(combined_filename)
    print(f"Saved combined dataset to: {combined_filename}")

    return country_data, combined_data


def analyze_data_completeness(country_data):
    """
    Analyze and print data completeness statistics
    """
    print("\nData Completeness Analysis:")
    print("-" * 40)

    for country, df in country_data.items():
        total_possible = len(df) * len(df.columns)
        total_actual = df.count().sum()
        completeness = (total_actual / total_possible) * 100

        print(f"\n{country}:")
        print(f"Overall completeness: {completeness:.1f}%")
        print("Indicators with most missing data:")
        missing = df.isnull().sum().sort_values(ascending=False).head()
        for indicator, count in missing.items():
            if count > 0:
                print(f"- {indicator}: {count} years missing")


def main():
    try:
        # Collect data
        country_data, combined_data = collect_worldbank_data()

        # Analyze completeness
        analyze_data_completeness(country_data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

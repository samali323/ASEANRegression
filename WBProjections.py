import pandas as pd
import requests
import os
from datetime import datetime

def get_projection_indicators():
    """
    Return dictionary of indicators to collect projection data for
    """
    return {
        # Economic Indicators
        'NY.GDP.MKTP.CD': 'GDP (current US$)',
        'NV.IND.TOTL.ZS': 'Industry (% of GDP)',
        'NV.IND.MANF.ZS': 'Manufacturing (% of GDP)',
        'BX.KLT.DINV.WD.GD.ZS': 'Foreign direct investment (% of GDP)',

        # Population Indicators
        'SP.POP.TOTL': 'Population, total',
        'SP.URB.TOTL': 'Urban population (total)',
        'SP.URB.GROW': 'Urban population growth',

        # Energy Indicators
        'EG.USE.ELEC.KH.PC': 'Electric power consumption (kWh per capita)',
        'EG.ELC.ACCS.ZS': 'Access to electricity (% of population)',
        'EG.USE.COMM.FO.ZS': 'Fossil fuel energy consumption (% of total)',

        # Additional Indicators from Previous Analysis
        'EG.CFT.ACCS.ZS': 'Access to clean fuels and technologies for cooking',
        'EG.RNW.TOTL.ZS': 'Renewable energy consumption (% of total final energy consumption)',
        'EG.IMP.CONS.ZS': 'Energy imports (% of energy use)',
        'SP.POP.GROW': 'Population growth rate',
        'NY.GNP.PCAP.CD': 'GNI per capita',
        'EN.ATM.GHGT.KT.CE': 'Total greenhouse gas emissions'
    }

def fetch_projection_data(country_code, indicator_code, start_year, end_year):
    """
    Fetch projection data from World Bank API

    Parameters:
    -----------
    country_code : str
        Country code
    indicator_code : str
        World Bank indicator code
    start_year : int
        Start year for projection
    end_year : int
        End year for projection

    Returns:
    --------
    pandas.Series
        Projection data for the indicator
    """
    base_url = "http://api.worldbank.org/v2/country"

    # Attempt to fetch projection data
    projection_url = f"{base_url}/{country_code}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}&source=57"

    try:
        response = requests.get(projection_url)

        if response.status_code == 200:
            data = response.json()

            if len(data) > 1 and data[1]:
                projection_values = {}
                for entry in data[1]:
                    if entry['value'] is not None:
                        projection_values[int(entry['date'])] = entry['value']

                return pd.Series(projection_values)

        return pd.Series()

    except Exception as e:
        print(f"Error fetching projection data for {country_code}, {indicator_code}: {str(e)}")
        return pd.Series()

def collect_projection_data(countries, start_year=2024, end_year=2035):
    """
    Collect projection data for specified countries

    Parameters:
    -----------
    countries : dict
        Dictionary of countries and their codes
    start_year : int, optional
        Start year for projection (default 2024)
    end_year : int, optional
        End year for projection (default 2035)

    Returns:
    --------
    dict
        Dictionary of projection datasets for each country
    """
    # Create output directory
    output_dir = 'projection_data'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize storage for projection data
    projection_datasets = {}

    # Get indicators to collect
    indicators = get_projection_indicators()

    # Timestamp for this data collection run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Collect projection data for each country
    for country_name, country_code in countries.items():
        print(f"\nCollecting projection data for {country_name} ({country_code})...")

        # Initialize DataFrame for projections
        proj_df = pd.DataFrame(index=range(start_year, end_year + 1))
        proj_df.index.name = 'Year'
        proj_df['Country'] = country_name

        # Collect each indicator's projection
        for indicator_code, indicator_name in indicators.items():
            print(f"  Collecting {indicator_name} projection...", end=' ')

            projection_series = fetch_projection_data(country_code, indicator_code, start_year, end_year)

            if not projection_series.empty:
                proj_df[indicator_name] = projection_series
                print(f"✓ ({len(projection_series)} years)")
            else:
                print("✗ No projection data")

        # Save country-specific projection data
        filename = f"{output_dir}/{country_name.replace(' ', '_')}_Projections_{timestamp}.csv"
        proj_df.to_csv(filename, index=True)

        projection_datasets[country_name] = proj_df

    # Create combined projection dataset
    combined_proj_df = pd.concat(list(projection_datasets.values()))
    combined_filename = f"{output_dir}/ASEAN_Combined_Projections_{timestamp}.csv"
    combined_proj_df.to_csv(combined_filename, index=True)

    return projection_datasets, combined_proj_df

def main():
    # ASEAN countries and their World Bank codes
    asean_countries = {
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

    # Collect projection data
    projection_datasets, combined_proj_df = collect_projection_data(asean_countries)

    # Analyze data completeness
    print("\nPROJECTION DATA COMPLETENESS:")
    for country, df in projection_datasets.items():
        completeness = df.notna().mean() * 100
        print(f"\n{country}:")
        print("Projection Data Availability:")
        for indicator, avail in completeness.items():
            print(f"  {indicator}: {avail:.1f}%")

if __name__ == "__main__":
    main()

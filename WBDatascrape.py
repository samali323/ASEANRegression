import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
import glob
import sys


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
    """Return dictionary of indicators to collect, organized by category"""
    return {
        # Population Indicators
        'SP.POP.TOTL': 'Population, total',
        'SP.POP.GROW': 'Population growth (annual %)',
        'SP.URB.TOTL.IN.ZS': 'Urban population (% of total)',
        'SP.RUR.TOTL.ZS': 'Rural population (% of total)',

        # Economic Indicators
        'NY.GDP.MKTP.CD': 'GDP (current US$)',
        'NY.GNP.PCAP.CD': 'GNI per capita',
        'NV.IND.TOTL.ZS': 'Industry (% of GDP)',
        'NV.IND.MANF.ZS': 'Manufacturing (% of GDP)',
        'BX.KLT.DINV.WD.GD.ZS': 'Foreign direct investment (% of GDP)',

        # Energy Demand and Consumption
        'EG.USE.PCAP.KG.OE': 'Energy use (kg of oil equivalent per capita)',
        'EG.USE.ELEC.KH.PC': 'Electric power consumption (kWh per capita)',
        'EG.USE.COMM.CL.ZS': 'Residential energy consumption (% of total)',

        # Energy Production & Mix
        'EG.ELC.PROD.KH': 'Electricity production (kWh)',
        'EG.ELC.RNEW.ZS': 'Renewable electricity output (% of total)',
        'EG.ELC.FOSL.ZS': 'Fossil fuel electricity production (%)',
        'EG.USE.COMM.FO.ZS': 'Fossil fuel energy consumption (% of total)',
        'EG.RNW.TOTL.ZS': 'Renewable energy consumption (% of total final energy consumption)',

        # Energy Infrastructure & Access
        'EG.ELC.ACCS.ZS': 'Access to electricity (% of population)',
        'EG.ELC.LOSS.ZS': 'Electric power transmission and distribution losses (% of output)',
        'EG.CFT.ACCS.ZS': 'Access to clean fuels and technologies for cooking',

        # Energy Trade & Dependency
        'EG.IMP.CONS.ZS': 'Energy imports (% of energy use)',

        # Energy Efficiency & Policy
        'EN.ATM.CO2E.PP.GD': 'CO2 intensity (kg per 2015 US$ GDP)',
        'EG.EGY.PRIM.PP.KD': 'Energy productivity (GDP per unit energy)',

        # Technology and Infrastructure
        'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)',
        'IT.CEL.SETS.P2': 'Mobile cellular subscriptions (per 100 people)',

        # Environmental and Pollution Indicators
        'EN.ATM.PM25.MC.M3': 'PM2.5 air pollution (micrograms per cubic meter)',
        'EN.ATM.CO2E.PC': 'CO2 emissions (metric tons per capita)',
        'EN.ATM.GHGT.KT.CE': 'Total greenhouse gas emissions'
    }


def get_historical_demand():
    """Return DataFrame with historical demand data"""
    # INSERT HISTORICAL DEMAND DATAFRAME HERE
    data =  {
  "Area": ["Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Brunei Darussalam", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Cambodia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Indonesia", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Lao People's Democratic Republic (the)", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Malaysia", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Myanmar", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Philippines (the)", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Singapore", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Thailand", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam", "Viet Nam"],
  "Year": ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
  "Value": ["2.54", "2.58", "2.71", "3.2", "3.26", "3.26", "3.3", "3.39", "3.42", "3.62", "3.79", "3.73", "3.93", "4.4", "4.5", "4.2", "4.27", "4.16", "4.3", "4.93", "5.74", "5.7", "5.6", "5.59", "0.33", "0.28", "0.58", "0.65", "0.78", "0.98", "1.22", "1.59", "1.76", "1.99", "2.56", "2.89", "3.57", "4.09", "4.91", "6.13", "7.31", "8.29", "10.13", "12.1", "12.78", "13.46", "15.45", "18.87", "98.04", "106.16", "113.04", "117.39", "124.78", "132.89", "138", "148.4", "155.9", "162.87", "177.56", "191.88", "208.96", "225.18", "237.55", "243.29", "258.24", "266.95", "285.29", "297.12", "293.37", "310.33", "334.34", "351.49", "1.04", "0.99", "1.03", "1.12", "0.83", "1.34", "1.74", "2.42", "2.25", "2.62", "2.92", "3.07", "3.9", "4.14", "4.74", "6.66", "6.7", "6.67", "7.51", "8.47", "10.59", "11.53", "11.95", "15.26", "69.88", "75.69", "80.84", "83.11", "95.54", "96.52", "101.67", "106.29", "113.35", "116.02", "124.89", "127.43", "134.17", "141.16", "147.48", "150.12", "155.99", "159.52", "169.09", "176.84", "172.46", "174.27", "182.25", "186.66", "5.1", "4.67", "5.07", "5.42", "5.61", "6.02", "6.18", "6.4", "6.62", "6.97", "8.62", "10.43", "11.16", "12.46", "14.39", "16.2", "15.74", "19.65", "20.21", "18.05", "17.54", "18.66", "23.05", "25.49", "45.3", "47.05", "48.46", "52.94", "55.94", "56.57", "56.78", "59.62", "60.82", "61.92", "67.73", "69.15", "72.89", "75.23", "77.23", "82.4", "90.79", "94.36", "99.74", "106.03", "101.75", "105.75", "111.52", "111.52", "31.67", "33.06", "34.66", "35.32", "36.82", "38.22", "39.49", "41.13", "41.67", "41.8", "45.36", "46", "46.94", "47.97", "49.3", "50.28", "51.59", "52.28", "52.97", "54.14", "53.07", "55.79", "57.1", "57.4", "98.34", "103.7", "111.06", "118.21", "127.25", "134.3", "141.35", "146.27", "147.26", "147.05", "163.51", "162.74", "177.58", "179.93", "183.95", "189.98", "198.21", "200.05", "203.34", "209.17", "203.41", "207.75", "213.78", "224.04", "26.56", "30.61", "35.8", "40.93", "46.2", "52.46", "58.87", "66.79", "74.17", "84.37", "96.36", "105.38", "116.61", "126.79", "142.69", "162.91", "180.5", "195.43", "218.8", "238.14", "245.51", "254.43", "267.43", "278.61"]
}

    df = pd.DataFrame(data)

    # Convert Year to int and Value to float
    df['Year'] = df['Year'].astype(int)
    df['Value'] = df['Value'].astype(float)

    # Rename columns to match our format
    df = df.rename(columns={
        'Area': 'Country',
        'Value': 'Demand (TWh)'
    })

    # Fix country names to match World Bank format
    country_name_fixes = {
        "Lao People's Democratic Republic (the)": "Lao PDR",
        "Philippines (the)": "Philippines",
        "Viet Nam": "Vietnam"
    }
    df['Country'] = df['Country'].replace(country_name_fixes)

    return df


def get_hardcoded_projections():
    """Return dictionary of known projections data"""
    return {
        'SP.URB.TOTL.IN.ZS': {  # Urban population (% of total)
            'BRN': {2024: 79.439, 2025: 79.726, 2026: 80.007, 2027: 80.285, 2028: 80.559, 2029: 80.828,
                    2030: 81.093, 2031: 81.353, 2032: 81.61, 2033: 81.863, 2034: 82.111, 2035: 82.355},
            'KHM': {2024: 26.035, 2025: 26.51, 2026: 26.995, 2027: 27.49, 2028: 27.995, 2029: 28.51,
                    2030: 29.034, 2031: 29.568, 2032: 30.112, 2033: 30.665, 2034: 31.227, 2035: 31.798},
            'IDN': {2024: 59.204, 2025: 59.828, 2026: 60.446, 2027: 61.056, 2028: 61.659, 2029: 62.254,
                    2030: 62.841, 2031: 63.42, 2032: 63.99, 2033: 64.551, 2034: 65.103, 2035: 65.646},
            'LAO': {2024: 38.906, 2025: 39.566, 2026: 40.229, 2027: 40.893, 2028: 41.559, 2029: 42.225,
                    2030: 42.891, 2031: 43.557, 2032: 44.222, 2033: 44.886, 2034: 45.548, 2035: 46.207},
            'MYS': {2024: 79.201, 2025: 79.67, 2026: 80.124, 2027: 80.562, 2028: 80.984, 2029: 81.392,
                    2030: 81.786, 2031: 82.165, 2032: 82.531, 2033: 82.883, 2034: 83.223, 2035: 83.549},
            'MMR': {2024: 32.47, 2025: 32.846, 2026: 33.239, 2027: 33.65, 2028: 34.079, 2029: 34.525,
                    2030: 34.99, 2031: 35.474, 2032: 35.975, 2033: 36.494, 2034: 37.032, 2035: 37.587},
            'PHL': {2024: 48.614, 2025: 48.958, 2026: 49.318, 2027: 49.695, 2028: 50.088, 2029: 50.498,
                    2030: 50.924, 2031: 51.365, 2032: 51.823, 2033: 52.295, 2034: 52.783, 2035: 53.285},
            'SGP': {2024: 100, 2025: 100, 2026: 100, 2027: 100, 2028: 100, 2029: 100,
                    2030: 100, 2031: 100, 2032: 100, 2033: 100, 2034: 100, 2035: 100},
            'THA': {2024: 54.32, 2025: 55.024, 2026: 55.721, 2027: 56.409, 2028: 57.088, 2029: 57.759,
                    2030: 58.42, 2031: 59.071, 2032: 59.713, 2033: 60.344, 2034: 60.965, 2035: 61.574},
            'VNM': {2024: 40.195, 2025: 40.909, 2026: 41.622, 2027: 42.334, 2028: 43.043, 2029: 43.751,
                    2030: 44.455, 2031: 45.155, 2032: 45.851, 2033: 46.543, 2034: 47.229, 2035: 47.909}
        },
        'SP.RUR.TOTL.ZS': {  # Rural population (% of total)
            'BRN': {2024: 20.561, 2025: 20.274, 2026: 19.993, 2027: 19.715, 2028: 19.441, 2029: 19.172,
                    2030: 18.907, 2031: 18.647, 2032: 18.39, 2033: 18.137, 2034: 17.889, 2035: 17.645},
            # Add data for other countries...
        },
        'SP.POP.TOTL': {  # Population, total
            'BRN': {2024: 462721, 2025: 466330, 2026: 469775, 2027: 473156, 2028: 476433, 2029: 479495,
                    2030: 482447, 2031: 485316, 2032: 487952, 2033: 490437, 2034: 492894, 2035: 495290},
            # Add data for other countries...
        }
    }


def fetch_wb_data(country_code, indicator, start_year, end_year):
    """Fetch data from World Bank API with better projection handling and hardcoded data"""
    base_url = "http://api.worldbank.org/v2/country"

    # Try different sources and parameters combinations
    sources = [
        {"source": "2", "params": ""},  # Default source
        {"source": "6", "params": "&Projection=Y"},  # Health Nutrition and Population Statistics
        {"source": "57", "params": "&Projection=Y"},  # WB population projections
        {"source": "1", "params": ""},  # World Development Indicators
        {"source": "11", "params": ""},  # Health Nutrition Population Statistics
        {"source": "12", "params": ""},  # Education Statistics
        {"source": "13", "params": ""},  # Millennium Development Goals
        {"source": "14", "params": ""},  # Poverty and Equity
        {"source": "15", "params": ""},  # Global Economic Monitor
        {"source": "16", "params": ""},  # IDA Results Measurement System
        {"source": "19", "params": ""},  # Gender Statistics
        {"source": "25", "params": ""},  # Jobs
        {"source": "27", "params": ""},  # Global Economic Prospects
        {"source": "28", "params": ""},  # Global Financial Development
        {"source": "30", "params": ""},  # Doing Business
        {"source": "31", "params": ""},  # Worldwide Governance Indicators
        {"source": "37", "params": ""},  # Universal Health Coverage
        {"source": "43", "params": ""},  # Sustainable Energy for All
        {"source": "57", "params": "&Projection=Y"}  # Population projections
    ]

    all_data = {}

    # First try API endpoints
    for source_config in sources:
        try:
            url = f"{base_url}/{country_code}/indicator/{indicator}?format=json&date={start_year}:{end_year}&source={source_config['source']}{source_config['params']}&per_page=100"
            print(f"  Trying source {source_config['source']}...", end=' ')

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    for entry in data[1]:
                        if entry['value'] is not None:
                            year = int(entry['date'])
                            # For projections, prefer newer source data
                            if year not in all_data or year >= 2023:
                                all_data[year] = entry['value']
                    print(f"Found {len(data[1])} entries")
                else:
                    print("No data")
            else:
                print(f"Error {response.status_code}")

        except Exception as e:
            print(f"Error: {str(e)}")
            continue

    # Check if we have hardcoded projections for this indicator and country
    hardcoded_data = get_hardcoded_projections()
    if indicator in hardcoded_data and country_code in hardcoded_data[indicator]:
        print(f"  Adding hardcoded projections for {indicator}...")
        for year, value in hardcoded_data[indicator][country_code].items():
            if year not in all_data:
                all_data[year] = value

    # Sort the data by year
    sorted_data = dict(sorted(all_data.items()))
    return pd.Series(sorted_data)


def collect_worldbank_data(start_year=2000, end_year=2035):
    """Collect historical and projected World Bank data"""
    countries = get_asean_countries()
    indicators = get_indicators()

    # Load historical demand data
    historical_demand = get_historical_demand()

    output_dir = 'worldbank_data'
    if os.path.exists(output_dir):
        # Remove existing files
        for file in glob.glob(os.path.join(output_dir, '*.csv')):
            os.remove(file)
    os.makedirs(output_dir, exist_ok=True)

    country_data = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Collecting World Bank data from {start_year} to {end_year}")
    print(f"Total indicators: {len(indicators)}")
    print(f"Total countries: {len(countries)}\n")

    for country_name, country_code in countries.items():
        print(f"\nCollecting data for {country_name} ({country_code})...")

        # Create base DataFrame with years
        years = list(range(start_year, end_year + 1))
        country_df = pd.DataFrame({'Year': years})

        # Add historical demand data for this country
        country_demand = historical_demand[historical_demand['Country'] == country_name]
        if not country_demand.empty:
            demand_series = pd.Series(index=country_demand['Year'], data=country_demand['Demand (TWh)'].values)
            country_df['Demand (TWh)'] = country_df['Year'].map(demand_series)

        # Collect other indicators
        for indicator_code, indicator_name in indicators.items():
            print(f"\nCollecting {indicator_name}...")
            series = fetch_wb_data(country_code, indicator_code, start_year, end_year)

            if not series.empty:
                country_df[indicator_name] = country_df['Year'].map(series)
                print(f"Total years with data: {len(series)}")
                if len(series) > 0:
                    print(f"Data range: {min(series.index)} to {max(series.index)}")
            else:
                print("No data available")

        # Save country data
        country_data[country_name] = country_df
        filename = f"{output_dir}/{country_name.replace(' ', '_')}_{timestamp}.csv"
        country_df.to_csv(filename, index=False)
        print(f"Saved to: {filename}")

        # Print coverage summary
        print(f"\nData coverage summary for {country_name}:")
        for column in country_df.columns:
            if column != 'Year':
                total_years = end_year - start_year + 1
                available_years = country_df[column].notna().sum()
                coverage = (available_years / total_years) * 100
                print(f"  {column}: {available_years}/{total_years} years ({coverage:.1f}%)")

    # Create combined dataset
    combined_data = pd.DataFrame()
    for country_name, df in country_data.items():
        df_copy = df.copy()
        df_copy['Country'] = country_name
        combined_data = pd.concat([combined_data, df_copy], ignore_index=True)

    # Save combined dataset
    combined_filename = f"{output_dir}/ASEAN_combined_{timestamp}.csv"
    combined_data.to_csv(combined_filename, index=False)
    print(f"\nSaved combined dataset to: {combined_filename}")

    return country_data, combined_data


if __name__ == "__main__":
    try:
        country_data, combined_data = collect_worldbank_data()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback

        traceback.print_exc()
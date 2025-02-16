import os
import glob
import pandas as pd


def get_asean_countries():
    """Return dictionary of ASEAN countries and their 3-letter codes"""
    return {
        'Brunei Darussalam': 'brn',
        'Cambodia': 'khm',
        'Indonesia': 'idn',
        'Lao PDR': 'lao',
        'Malaysia': 'mys',
        'Myanmar': 'mmr',
        'Philippines': 'phl',
        'Singapore': 'sgp',
        'Thailand': 'tha',
        'Vietnam': 'vnm'
    }


def get_indicators():
    """Return dictionary of indicators to collect with their codes"""
    return {
        # Economic Indicators
        'ny_gdp_mktp_cd': 'GDP (current US$)',
        'ny_gnp_pcap_cd': 'GNI per capita',
        'nv_ind_totl_zs': 'Industry (% of GDP)',
        'nv_ind_manf_zs': 'Manufacturing (% of GDP)',
        'bx_klt_dinv_wd_gd_zs': 'Foreign direct investment (% of GDP)',

        # Population Indicators
        'sp_pop_totl': 'Population, total',
        'sp_urb_totl_in_zs': 'Urban population (% of total)',
        'sp_pop_grow': 'Population growth (annual %)',

        # Energy Indicators
        'eg_use_pcap_kg_oe': 'Energy use (kg of oil equivalent per capita)',
        'eg_use_elec_kh_pc': 'Electric power consumption (kWh per capita)',
        'eg_elc_accs_zs': 'Access to electricity (% of population)',
        'eg_use_comm_fo_zs': 'Fossil fuel energy consumption (% of total)',

        # Technology Indicators
        'it_net_user_zs': 'Individuals using the Internet (% of population)',
        'it_cel_sets_p2': 'Mobile cellular subscriptions (per 100 people)',

        # Environmental Indicators
        'en_atm_pm25_mc_m3': 'PM2.5 air pollution (micrograms per cubic meter)',
        'en_ghg_co2_pc_ce_ar5': 'CO2 emissions (metric tons per capita)',
        'en_ghg_all_pc_ce_ar5': 'Total greenhouse gas emissions (per capita)'
    }


def process_open_numbers_data(base_path):
    """
    Process Open Numbers World Development Indicators data

    Parameters:
    -----------
    base_path : str
        Base directory containing the WDI data files

    Returns:
    --------
    dict
        Dictionary of DataFrames for each ASEAN country
    """
    # Get ASEAN countries and their codes
    asean_countries = get_asean_countries()
    indicators = get_indicators()

    # Create output directory
    output_dir = os.path.join(base_path, 'processed_asean_data')
    os.makedirs(output_dir, exist_ok=True)

    # Prepare storage for processed data
    country_data = {}

    # Find all datapoint files
    all_files = glob.glob(os.path.join(base_path, '**', 'ddf--datapoints--*--by--geo--time.csv'), recursive=True)

    # Process each indicator
    for indicator_code, indicator_name in indicators.items():
        # Find matching file
        matching_files = [f for f in all_files if f.endswith(f'--datapoints--{indicator_code}--by--geo--time.csv')]

        if not matching_files:
            print(f"No file found for indicator: {indicator_code} - {indicator_name}")
            continue

        # Read the file
        try:
            df = pd.read_csv(matching_files[0])
        except Exception as e:
            print(f"Error reading file for {indicator_code}: {e}")
            continue

        # Process data for each ASEAN country
        for country_name, country_code in asean_countries.items():
            # Filter data for this country
            country_rows = df[df['geo'] == country_code].copy()

            if country_rows.empty:
                print(f"No data found for {country_name} in {indicator_name}")
                continue

            # Rename columns
            country_rows = country_rows.rename(columns={
                'time': 'Year',
                country_rows.columns[2]: indicator_name
            })

            # Initialize country DataFrame if not exists
            if country_name not in country_data:
                year_range = range(
                    country_rows['Year'].min(),
                    country_rows['Year'].max() + 1
                )
                country_data[country_name] = pd.DataFrame({'Year': year_range})

            # Merge data
            country_data[country_name] = country_data[country_name].merge(
                country_rows[['Year', indicator_name]],
                on='Year',
                how='left'
            )

    # Save individual country files and create combined dataset
    combined_data = pd.DataFrame()

    for country_name, df in country_data.items():
        # Add country column
        df['Country'] = country_name

        # Save individual country file
        output_filename = os.path.join(output_dir, f'{country_name.replace(" ", "_")}_open_numbers_data.csv')
        df.to_csv(output_filename, index=False)
        print(f"Saved data for {country_name} to {output_filename}")

        # Append to combined dataset
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Save combined dataset
    combined_output_filename = os.path.join(output_dir, 'ASEAN_combined_open_numbers_data.csv')
    combined_data.to_csv(combined_output_filename, index=False)
    print(f"\nSaved combined dataset to {combined_output_filename}")

    return country_data


def main():
    # Path to the World Development Indicators data
    base_path = r'C:\Users\samal\Documents\ddf--open_numbers--world_development_indicators-master'

    # Process the data
    processed_data = process_open_numbers_data(base_path)

    # Print data availability summary
    for country, df in processed_data.items():
        print(f"\n{country} Data Summary:")
        print(f"  Years covered: {df['Year'].min()} to {df['Year'].max()}")
        print("  Data availability:")
        for column in df.columns:
            if column not in ['Year', 'Country']:
                availability = (df[column].notna().sum() / len(df)) * 100
                print(f"    {column}: {availability:.1f}%")


if __name__ == "__main__":
    main()
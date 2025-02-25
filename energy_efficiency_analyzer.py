import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
def debug_print_dataframe_info(df, label="DataFrame"):
    """Print helpful debugging information about a dataframe"""
    print(f"\n--- DEBUG INFO: {label} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Years range: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Countries: {df['Country'].unique().tolist() if 'Country' in df.columns else 'No Country column'}")

    # Print first few rows
    print(f"\nFirst 3 rows:")
    print(df.head(3))

    # Print sample of future data (2024+)
    future_data = df[df['Year'] >= 2024] if 'Year' in df.columns else pd.DataFrame()
    if not future_data.empty:
        print(f"\nSample of future data (2024+):")
        print(future_data.head(3))

    # Check for key indicators
    key_indicators = ['Demand (TWh)', 'GDP (current US$)', 'CO2 emissions (metric tons per capita)', 'Population, total']
    print("\nKey indicators availability:")
    for indicator in key_indicators:
        variants = [
            indicator,
            f"{indicator}_Prophet",
            f"{indicator}_VAR_Forecast"
        ]
        found = False
        for var in variants:
            if var in df.columns:
                non_null = df[var].notna().sum()
                print(f"  - {var}: {non_null}/{len(df)} non-null values")
                if non_null > 0 and future_data.shape[0] > 0:
                    future_non_null = future_data[var].notna().sum()
                    print(f"    Future values (2024+): {future_non_null}/{future_data.shape[0]} non-null")
                found = True
        if not found:
            print(f"  - {indicator}: Not found in any form")
    print("--- END DEBUG INFO ---\n")

# Now replace the _generate_scenarios method with this improved version:

def _generate_scenarios(self, country, forecasts):
    """
    Generate energy efficiency scenarios with detailed debugging

    Parameters:
    -----------
    country : str
        Country name
    forecasts : DataFrame
        Combined forecast data

    Returns:
    --------
    dict
        Dictionary of scenario results
    """
    print(f"  Generating efficiency scenarios for {country}")

    # Print debug info about the incoming data
    debug_print_dataframe_info(forecasts, f"Input data for {country}")

    # Get baseline data
    baseline = forecasts.copy()

    # Filter for future years only (2024-2035)
    future_data = baseline[baseline['Year'] >= 2024].copy()
    if future_data.empty:
        raise ValueError(f"No future data (2024+) found for {country}")

    # Debug future data
    debug_print_dataframe_info(future_data, f"Future data for {country}")

    # Define the relevant indicators and column mapping
    indicators = {
        'demand': 'Demand (TWh)',
        'gdp': 'GDP (current US$)',
        'co2': 'CO2 emissions (metric tons per capita)',
        'population': 'Population, total'
    }

    # Helper function to find the best available column
    def find_best_column(base_indicator):
        variants = [
            f"{base_indicator}_Prophet",  # First try Prophet
            base_indicator,               # Then try direct column
            f"{base_indicator}_Source"    # Then try source indicator
        ]

        # Also check for any column that contains the base indicator name
        all_matching = [col for col in future_data.columns if base_indicator in col]
        variants.extend([col for col in all_matching if col not in variants])

        for variant in variants:
            if variant in future_data.columns and future_data[variant].notna().any():
                print(f"    Using column '{variant}' for {base_indicator}")
                return variant

        # If nothing found with non-null values, try even empty columns
        for variant in variants:
            if variant in future_data.columns:
                print(f"    WARNING: Using column '{variant}' for {base_indicator} but it contains all null values")
                return variant

        # If we get here, we couldn't find any suitable column
        print(f"    ERROR: No column found for {base_indicator}")
        print(f"    Available columns: {future_data.columns.tolist()}")
        raise ValueError(f"No suitable column found for {base_indicator}")

    # Get column names for each indicator
    try:
        demand_col = find_best_column(indicators['demand'])
        gdp_col = find_best_column(indicators['gdp'])
        co2_col = find_best_column(indicators['co2'])
        pop_col = find_best_column(indicators['population'])

        # Print summary of selected columns
        print("\n  Using these columns for analysis:")
        print(f"    Demand: {demand_col}")
        print(f"    GDP: {gdp_col}")
        print(f"    CO2: {co2_col}")
        print(f"    Population: {pop_col}")

        # Extract the data
        demand_data = future_data[demand_col].copy()
        gdp_data = future_data[gdp_col].copy()
        co2_data = future_data[co2_col].copy()
        pop_data = future_data[pop_col].copy()

        # Check for nulls and print warnings
        for name, series in [("Demand", demand_data), ("GDP", gdp_data),
                             ("CO2", co2_data), ("Population", pop_data)]:
            null_count = series.isna().sum()
            if null_count > 0:
                print(f"    WARNING: {name} has {null_count} null values out of {len(series)}")

                # Fill nulls with mean or last valid value
                if not series.isna().all():
                    if series.isna().all():
                        print(f"    ERROR: {name} data is all null, cannot proceed")
                        raise ValueError(f"{name} data contains all null values")

                    # Use forward fill then backward fill to handle nulls
                    series_filled = series.fillna(method='ffill').fillna(method='bfill')
                    if series_filled.isna().any():
                        # If still has nulls, use mean
                        mean_val = series.mean()
                        series_filled = series.fillna(mean_val)
                        print(f"    Filling null values with mean: {mean_val}")
                    else:
                        print(f"    Filled null values using forward/backward fill")

                    # Replace the original series
                    if name == "Demand":
                        demand_data = series_filled
                    elif name == "GDP":
                        gdp_data = series_filled
                    elif name == "CO2":
                        co2_data = series_filled
                    elif name == "Population":
                        pop_data = series_filled

        # Use the processed data
        baseline_values = {
            'demand': demand_data,
            'co2': co2_data,
            'gdp': gdp_data,
            'population': pop_data
        }

        # Print data preview
        print("\n  Data preview:")
        for key, data in baseline_values.items():
            print(f"    {key.capitalize()}: min={data.min()}, max={data.max()}, mean={data.mean()}")

        # Include climate change impacts (BAU scenario)
        years = future_data['Year'].values
        base_year = min(years)
        bau_demand = baseline_values['demand'].copy()

        # Apply climate factors
        for i, year in enumerate(years):
            years_passed = year - base_year
            # Apply compounding climate factor to demand
            climate_multiplier = (1 + self.climate_factors['cooling_demand_increase']) ** years_passed
            bau_demand.iloc[i] *= climate_multiplier

        # Apply modest CO2 increase due to climate change (less efficiency at higher temps)
        climate_affected_co2 = baseline_values['co2'].values * np.array(
            [(1 + 0.005) ** (y - base_year) for y in years]
        )

        # Initialize results dictionary
        results = {
            'years': years,
            'baseline': {
                'demand': baseline_values['demand'].values,
                'co2': baseline_values['co2'].values,
                'gdp': baseline_values['gdp'].values,
                'population': baseline_values['population'].values
            },
            'bau': {  # Business as usual with climate impacts
                'demand': bau_demand.values,
                'co2': climate_affected_co2
            },
            'scenarios': {}
        }

        # Generate scenarios
        for scenario, params in self.efficiency_scenarios.items():
            demand_reduction = params['demand_reduction']
            co2_reduction = params['co2_reduction']
            implementation_cost = params['implementation_cost']

            # Calculate values with efficiency improvements
            improved_demand = bau_demand.values * (1 - demand_reduction)
            improved_co2 = baseline_values['co2'].values * (1 - co2_reduction)

            # Calculate implementation costs
            implementation_costs = baseline_values['gdp'].values * implementation_cost

            # Print scenario summary
            print(f"\n  {scenario.capitalize()} scenario:")
            print(f"    Demand reduction: {demand_reduction*100:.1f}%")
            print(f"    CO2 reduction: {co2_reduction*100:.1f}%")
            print(f"    Implementation cost: {implementation_cost*100:.1f}% of GDP")

            # Store results
            results['scenarios'][scenario] = {
                'demand': improved_demand,
                'co2': improved_co2,
                'implementation_cost': implementation_costs,
                'params': params
            }

        return results

    except Exception as e:
        print(f"\n  ERROR in scenario generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Also update the _calculate_benefits method for better error handling:

def _calculate_benefits(self, country, forecasts, results):
    """
    Calculate economic and environmental benefits of efficiency scenarios with improved error handling
    """
    print(f"  Calculating benefits of efficiency improvements for {country}")

    try:
        benefits = {
            'economic': {},
            'environmental': {},
            'social': {}
        }

        years = results['years']

        # Energy cost assumptions (USD per MWh)
        energy_cost = {
            'current': 100,
            'projected': [100 * (1 + 0.02) ** (y - min(years)) for y in years]  # 2% annual increase
        }

        # Environmental damage cost (USD per ton of CO2)
        env_damage_cost = {
            'current': 40,
            'projected': [40 * (1 + 0.03) ** (y - min(years)) for y in years]  # 3% annual increase
        }

        # Calculate benefits for each scenario
        for scenario, data in results['scenarios'].items():
            # Economic benefits
            energy_savings_twh = results['bau']['demand'] - data['demand']
            energy_cost_savings = np.array([energy_savings_twh[i] * 1000 * energy_cost['projected'][i]
                                            for i in range(len(years))])

            # Environmental benefits
            co2_reduction = results['baseline']['co2'] - data['co2']
            total_co2_reduction = co2_reduction * results['baseline']['population']
            env_damage_avoided = np.array([total_co2_reduction[i] * env_damage_cost['projected'][i]
                                           for i in range(len(years))])

            # Social benefits (job creation, health improvements)
            job_creation = np.array([data['implementation_cost'][i] * 0.00001 for i in range(len(years))])
            health_benefits = env_damage_avoided * 0.3  # 30% of environmental benefits are health-related

            # Net economic benefit
            net_benefit = energy_cost_savings + env_damage_avoided - data['implementation_cost']

            # Payback period calculation
            cumulative_cost = np.cumsum(data['implementation_cost'])
            cumulative_benefit = np.cumsum(energy_cost_savings + env_damage_avoided)

            payback_period = None
            for i in range(len(years)):
                if cumulative_benefit[i] >= cumulative_cost[i]:
                    payback_period = years[i] - min(years)
                    break

            # ROI calculation (Return on Investment)
            total_investment = np.sum(data['implementation_cost'])
            total_benefit = np.sum(energy_cost_savings + env_damage_avoided)

            # Check for division by zero
            if total_investment > 0:
                roi = (total_benefit - total_investment) / total_investment
            else:
                print(f"    WARNING: Implementation cost is zero or negative for {scenario}, setting ROI to 0")
                roi = 0

            # Print benefit summary
            print(f"\n  {scenario.capitalize()} scenario benefits:")
            print(f"    Total energy savings: {np.sum(energy_savings_twh):.2f} TWh")
            print(f"    Total energy cost savings: ${np.sum(energy_cost_savings)/1e9:.2f} billion")
            print(f"    Total CO2 reduction: {np.sum(total_co2_reduction)/1e6:.2f} million tons")
            print(f"    ROI: {roi*100:.1f}%")
            print(f"    Payback period: {payback_period:.1f} years" if payback_period else "    Payback period: Beyond timeframe")

            # Store results
            benefits['economic'][scenario] = {
                'energy_savings_twh': energy_savings_twh,
                'energy_cost_savings': energy_cost_savings,
                'implementation_cost': data['implementation_cost'],
                'net_benefit': net_benefit,
                'payback_period': payback_period,
                'roi': roi
            }

            benefits['environmental'][scenario] = {
                'co2_reduction_per_capita': co2_reduction,
                'total_co2_reduction': total_co2_reduction,
                'env_damage_avoided': env_damage_avoided
            }

            benefits['social'][scenario] = {
                'job_creation': job_creation,
                'health_benefits': health_benefits
            }

        return benefits

    except Exception as e:
        print(f"\n  ERROR in benefits calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a minimal valid structure to avoid further errors
        default_benefits = {
            'economic': {scenario: {'roi': 0, 'payback_period': None, 'net_benefit': np.zeros(len(years)),
                                    'energy_savings_twh': np.zeros(len(years)), 'energy_cost_savings': np.zeros(len(years)),
                                    'implementation_cost': np.zeros(len(years))}
                         for scenario in results['scenarios']},
            'environmental': {scenario: {'co2_reduction_per_capita': np.zeros(len(years)),
                                         'total_co2_reduction': np.zeros(len(years)),
                                         'env_damage_avoided': np.zeros(len(years))}
                              for scenario in results['scenarios']},
            'social': {scenario: {'job_creation': np.zeros(len(years)), 'health_benefits': np.zeros(len(years))}
                       for scenario in results['scenarios']}
        }
        return default_benefits
class EnergyEfficiencyAnalyzer:
    """
    Analyzes energy efficiency potential, carbon savings, and economic benefits
    based on ASEAN forecaster outputs.
    """

    def __init__(self, base_forecaster, efficiency_scenarios=None):
        """
        Initialize with a base forecaster object

        Parameters:
        -----------
        base_forecaster : ASEANForecaster
            The base forecaster object with data and forecasts
        efficiency_scenarios : dict, optional
            Dictionary of efficiency improvement scenarios
        """
        self.forecaster = base_forecaster
        self.data = base_forecaster.data

        # Default efficiency improvement scenarios if none provided
        self.efficiency_scenarios = efficiency_scenarios or {
            'low': {
                'demand_reduction': 0.10,  # 10% reduction in energy demand
                'co2_reduction': 0.15,     # 15% reduction in CO2 emissions
                'implementation_cost': 0.02  # 2% of GDP
            },
            'medium': {
                'demand_reduction': 0.20,  # 20% reduction in energy demand
                'co2_reduction': 0.30,     # 30% reduction in CO2 emissions
                'implementation_cost': 0.04  # 4% of GDP
            },
            'high': {
                'demand_reduction': 0.35,  # 35% reduction in energy demand
                'co2_reduction': 0.50,     # 50% reduction in CO2 emissions
                'implementation_cost': 0.07  # 7% of GDP
            }
        }

        # Climate change impact factors (conservative estimates)
        self.climate_factors = {
            'cooling_demand_increase': 0.015,  # 1.5% increase per year in cooling demand
            'temperature_increase': 0.02,      # 2% increase in average temperature per year
            'extreme_weather_cost': 0.005      # 0.5% of GDP impact from extreme weather
        }

        # Carbon credit pricing (USD per ton of CO2)
        self.carbon_credit_pricing = {
            'current': 15,         # Current average price
            'projected_2030': 40,  # Projected 2030 price
            'projected_2035': 75   # Projected 2035 price
        }

        # Create output directories
        self._create_output_dirs()

    def _create_output_dirs(self):
        """Create necessary output directories"""
        if self.forecaster.selected_country:
            countries = [self.forecaster.selected_country]
        else:
            countries = self.data['Country'].unique()

        for country in countries:
            country_str = str(country)
            os.makedirs(os.path.join('output', country_str, 'efficiency_analysis'), exist_ok=True)
            os.makedirs(os.path.join('output', country_str, 'efficiency_analysis', 'plots'), exist_ok=True)

    def analyze_country(self, country):
        """
        Perform energy efficiency analysis for a specific country

        Parameters:
        -----------
        country : str
            Country name
        """
        print(f"\nAnalyzing energy efficiency potential for {country}")

        # Convert country to string for consistency
        country = str(country)

        # Load combined forecast data
        forecast_file = os.path.join('output', country, f'{country}_combined_forecast.csv')

        if not os.path.exists(forecast_file):
            print(f"  Error: No forecast data found for {country}. Run forecasts first.")
            return None

        try:
            forecasts = pd.read_csv(forecast_file)

            # Check if required indicators exist (in any form - original or forecast)
            required_base_indicators = ['Demand (TWh)', 'GDP (current US$)', 'CO2 emissions (metric tons per capita)', 'Population, total']

            # Check for basic existence of indicators (in any form)
            missing_indicators = []
            for indicator in required_base_indicators:
                if indicator not in forecasts.columns and not any(col.startswith(indicator + '_') for col in forecasts.columns):
                    missing_indicators.append(indicator)

            if missing_indicators:
                print(f"  Error: Missing required indicators: {missing_indicators}")
                return None

            # Generate scenarios
            results = self._generate_scenarios(country, forecasts)

            # Calculate benefits
            benefits = self._calculate_benefits(country, forecasts, results)

            # Calculate carbon credits
            carbon_credits = self._calculate_carbon_credits(country, forecasts, results)

            # Generate visualizations
            self._create_visualizations(country, forecasts, results, benefits, carbon_credits)

            # Generate report
            self._generate_report(country, forecasts, results, benefits, carbon_credits)

            return results

        except Exception as e:
            print(f"  Error analyzing {country}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_scenarios(self, country, forecasts):
        """
        Generate energy efficiency scenarios

        Parameters:
        -----------
        country : str
            Country name
        forecasts : DataFrame
            Combined forecast data

        Returns:
        --------
        dict
            Dictionary of scenario results
        """
        print(f"  Generating efficiency scenarios for {country}")

        # Get baseline data (use Prophet forecasts if available, otherwise VAR)
        baseline = forecasts.copy()

        # Filter for future years only (2024-2035)
        future_data = baseline[baseline['Year'] >= 2024].copy()

        # Get the relevant indicators (checking for availability)
        indicators = {
            'demand': 'Demand (TWh)',
            'gdp': 'GDP (current US$)',
            'co2': 'CO2 emissions (metric tons per capita)',
            'population': 'Population, total'
        }

        # Function to get the best available column for each indicator
        def get_best_column(base_indicator):
            # Try Prophet first
            prophet_col = f"{base_indicator}_Prophet"
            if prophet_col in future_data.columns:
                return prophet_col

            # Then try VAR/original
            if base_indicator in future_data.columns:
                return base_indicator

            # If none are found, raise an error
            available_cols = [col for col in future_data.columns if base_indicator in col]
            if available_cols:
                # Return the first available variant
                print(f"  Note: Using {available_cols[0]} instead of {base_indicator}")
                return available_cols[0]
            else:
                raise ValueError(f"No column found for {base_indicator}")

        # Get best available columns
        demand_col = get_best_column(indicators['demand'])
        co2_col = get_best_column(indicators['co2'])
        gdp_col = get_best_column(indicators['gdp'])
        population_col = get_best_column(indicators['population'])

        # Extract baseline values using the best available columns
        baseline_values = {
            'demand': future_data[demand_col],
            'co2': future_data[co2_col],
            'gdp': future_data[gdp_col],
            'population': future_data[population_col]
        }

        # Include climate change impacts (BAU scenario)
        years = future_data['Year'].values
        base_year = min(years)
        bau_demand = baseline_values['demand'].copy()

        for i, year in enumerate(years):
            years_passed = year - base_year
            # Apply compounding climate factor
            climate_multiplier = (1 + self.climate_factors['cooling_demand_increase']) ** years_passed
            bau_demand.iloc[i] *= climate_multiplier

        # Initialize results dictionary
        results = {
            'years': years,
            'baseline': {
                'demand': baseline_values['demand'].values,
                'co2': baseline_values['co2'].values,
                'gdp': baseline_values['gdp'].values,
                'population': baseline_values['population'].values
            },
            'bau': {  # Business as usual with climate impacts
                'demand': bau_demand.values,
                'co2': baseline_values['co2'].values * np.array([(1 + 0.005) ** (y - base_year) for y in years])
            },
            'scenarios': {}
        }

        # Generate scenarios
        for scenario, params in self.efficiency_scenarios.items():
            demand_reduction = params['demand_reduction']
            co2_reduction = params['co2_reduction']
            implementation_cost = params['implementation_cost']

            # Calculate values with efficiency improvements
            improved_demand = bau_demand.values * (1 - demand_reduction)
            improved_co2 = baseline_values['co2'].values * (1 - co2_reduction)

            # Calculate implementation costs
            implementation_costs = baseline_values['gdp'].values * implementation_cost

            # Store results
            results['scenarios'][scenario] = {
                'demand': improved_demand,
                'co2': improved_co2,
                'implementation_cost': implementation_costs,
                'params': params
            }

        return results

    def _calculate_benefits(self, country, forecasts, results):
        """
        Calculate economic and environmental benefits of efficiency scenarios

        Parameters:
        -----------
        country : str
            Country name
        forecasts : DataFrame
            Combined forecast data
        results : dict
            Scenario results

        Returns:
        --------
        dict
            Benefits calculations
        """
        print(f"  Calculating benefits of efficiency improvements for {country}")

        benefits = {
            'economic': {},
            'environmental': {},
            'social': {}
        }

        years = results['years']

        # Energy cost assumptions (USD per MWh)
        energy_cost = {
            'current': 100,
            'projected': [100 * (1 + 0.02) ** (y - min(years)) for y in years]  # 2% annual increase
        }

        # Environmental damage cost (USD per ton of CO2)
        env_damage_cost = {
            'current': 40,
            'projected': [40 * (1 + 0.03) ** (y - min(years)) for y in years]  # 3% annual increase
        }

        # Calculate benefits for each scenario
        for scenario, data in results['scenarios'].items():
            # Economic benefits
            energy_savings_twh = results['bau']['demand'] - data['demand']
            energy_cost_savings = np.array([energy_savings_twh[i] * 1000 * energy_cost['projected'][i]
                                            for i in range(len(years))])

            # Environmental benefits
            co2_reduction = results['baseline']['co2'] - data['co2']
            total_co2_reduction = co2_reduction * results['baseline']['population']
            env_damage_avoided = np.array([total_co2_reduction[i] * env_damage_cost['projected'][i]
                                           for i in range(len(years))])

            # Social benefits (job creation, health improvements)
            job_creation = np.array([data['implementation_cost'][i] * 0.00001 for i in range(len(years))])
            health_benefits = env_damage_avoided * 0.3  # 30% of environmental benefits are health-related

            # Net economic benefit
            net_benefit = energy_cost_savings + env_damage_avoided - data['implementation_cost']

            # Payback period calculation
            cumulative_cost = np.cumsum(data['implementation_cost'])
            cumulative_benefit = np.cumsum(energy_cost_savings + env_damage_avoided)
            payback_period = None
            for i in range(len(years)):
                if cumulative_benefit[i] >= cumulative_cost[i]:
                    payback_period = years[i] - min(years)
                    break

            # ROI calculation (Return on Investment)
            total_investment = np.sum(data['implementation_cost'])
            total_benefit = np.sum(energy_cost_savings + env_damage_avoided)
            roi = (total_benefit - total_investment) / total_investment if total_investment > 0 else 0

            # Store results
            benefits['economic'][scenario] = {
                'energy_savings_twh': energy_savings_twh,
                'energy_cost_savings': energy_cost_savings,
                'implementation_cost': data['implementation_cost'],
                'net_benefit': net_benefit,
                'payback_period': payback_period,
                'roi': roi
            }

            benefits['environmental'][scenario] = {
                'co2_reduction_per_capita': co2_reduction,
                'total_co2_reduction': total_co2_reduction,
                'env_damage_avoided': env_damage_avoided
            }

            benefits['social'][scenario] = {
                'job_creation': job_creation,
                'health_benefits': health_benefits
            }

        return benefits

    def _calculate_carbon_credits(self, country, forecasts, results):
        """
        Calculate potential carbon credits from efficiency improvements

        Parameters:
        -----------
        country : str
            Country name
        forecasts : DataFrame
            Combined forecast data
        results : dict
            Scenario results

        Returns:
        --------
        dict
            Carbon credit calculations
        """
        print(f"  Calculating potential carbon credits for {country}")

        carbon_credits = {}
        years = results['years']

        # Define carbon price trajectory
        start_price = self.carbon_credit_pricing['current']
        end_price = self.carbon_credit_pricing['projected_2035']

        # Linear interpolation for prices
        carbon_prices = []
        for year in years:
            if year <= 2030:
                # Linear increase from current to 2030 price
                progress = (year - 2024) / (2030 - 2024)
                price = start_price + progress * (self.carbon_credit_pricing['projected_2030'] - start_price)
            else:
                # Linear increase from 2030 to 2035 price
                progress = (year - 2030) / (2035 - 2030)
                price = self.carbon_credit_pricing['projected_2030'] + progress * (end_price - self.carbon_credit_pricing['projected_2030'])
            carbon_prices.append(price)

        for scenario, data in results['scenarios'].items():
            co2_reduction = results['baseline']['co2'] - data['co2']
            total_co2_reduction = co2_reduction * results['baseline']['population']

            # Calculate carbon credit value
            carbon_credit_value = np.array([total_co2_reduction[i] * carbon_prices[i]
                                            for i in range(len(years))])

            # Calculate cumulative credit generation
            cumulative_credits = np.cumsum(total_co2_reduction)
            cumulative_value = np.cumsum(carbon_credit_value)

            # Calculate aggregation potential (combining small projects into larger portfolios)
            aggregation_potential = carbon_credit_value * 1.2  # 20% premium for aggregated projects

            carbon_credits[scenario] = {
                'annual_credits': total_co2_reduction,
                'annual_value': carbon_credit_value,
                'cumulative_credits': cumulative_credits,
                'cumulative_value': cumulative_value,
                'aggregation_potential': aggregation_potential,
                'carbon_prices': carbon_prices
            }

        return carbon_credits

    def _create_visualizations(self, country, forecasts, results, benefits, carbon_credits):
        """
        Create visualizations for energy efficiency analysis

        Parameters:
        -----------
        country : str
            Country name
        forecasts : DataFrame
            Combined forecast data
        results : dict
            Scenario results
        benefits : dict
            Benefits calculations
        carbon_credits : dict
            Carbon credit calculations
        """
        print(f"  Creating visualizations for {country}")

        output_dir = os.path.join('output', country, 'efficiency_analysis', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        years = results['years']

        # 1. Energy Demand Scenarios
        plt.figure(figsize=(12, 8))
        plt.plot(years, results['bau']['demand'], 'r-', linewidth=3, label='Business As Usual (with climate change)')
        plt.plot(years, results['baseline']['demand'], 'k--', linewidth=2, label='Baseline Forecast (without climate change)')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (scenario, data) in enumerate(results['scenarios'].items()):
            plt.plot(years, data['demand'], marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Efficiency Scenario")

        plt.title(f"{country} - Energy Demand Scenarios with Efficiency Improvements", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Energy Demand (TWh)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_demand_scenarios.png'), dpi=300)
        plt.close()

        # 2. CO2 Emissions Scenarios
        plt.figure(figsize=(12, 8))
        plt.plot(years, results['bau']['co2'], 'r-', linewidth=3, label='Business As Usual (with climate change)')
        plt.plot(years, results['baseline']['co2'], 'k--', linewidth=2, label='Baseline Forecast')

        for i, (scenario, data) in enumerate(results['scenarios'].items()):
            plt.plot(years, data['co2'], marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Efficiency Scenario")

        plt.title(f"{country} - CO2 Emissions Scenarios with Efficiency Improvements", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'co2_emissions_scenarios.png'), dpi=300)
        plt.close()

        # 3. Economic Benefits
        plt.figure(figsize=(12, 8))

        bar_width = 0.25
        x = np.arange(len(years))

        for i, (scenario, data) in enumerate(benefits['economic'].items()):
            plt.bar(x + i*bar_width, data['net_benefit'] / 1e9, width=bar_width,
                    label=f"{scenario.capitalize()} Scenario", alpha=0.7)

        plt.title(f"{country} - Net Economic Benefits of Energy Efficiency", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Net Benefit (Billion USD)", fontsize=12)
        plt.xticks(x + bar_width, years)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'economic_benefits.png'), dpi=300)
        plt.close()

        # 4. Carbon Credits Value
        plt.figure(figsize=(12, 8))

        for i, (scenario, data) in enumerate(carbon_credits.items()):
            plt.plot(years, data['annual_value'] / 1e6, marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Scenario")

        plt.title(f"{country} - Annual Carbon Credit Value", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Carbon Credit Value (Million USD)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'carbon_credit_value.png'), dpi=300)
        plt.close()

        # 5. ROI and Payback Period Comparison
        plt.figure(figsize=(12, 8))

        scenarios = list(benefits['economic'].keys())
        rois = [benefits['economic'][s]['roi'] * 100 for s in scenarios]

        plt.bar(scenarios, rois, color=colors[:len(scenarios)])

        plt.title(f"{country} - Return on Investment for Energy Efficiency Scenarios", fontsize=15)
        plt.xlabel("Efficiency Scenario", fontsize=12)
        plt.ylabel("ROI (%)", fontsize=12)

        # Add payback period labels
        for i, scenario in enumerate(scenarios):
            payback = benefits['economic'][scenario]['payback_period']
            if payback:
                plt.text(i, rois[i] + 5, f"Payback: {payback:.1f} years",
                         ha='center', va='bottom', fontsize=12)
            else:
                plt.text(i, rois[i] + 5, "No payback\nwithin timeframe",
                         ha='center', va='bottom', fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roi_comparison.png'), dpi=300)
        plt.close()

        # 6. NDC Contribution
        plt.figure(figsize=(12, 8))

        # Estimate potential NDC contribution (example reduction targets)
        # Simplified assumption: 30% reduction from BAU by 2030
        bau_2030_index = np.where(years == 2030)[0][0] if 2030 in years else -1

        if bau_2030_index >= 0:
            bau_2030_co2 = results['bau']['co2'][bau_2030_index]
            ndc_target = bau_2030_co2 * 0.7  # 30% reduction

            plt.axhline(y=ndc_target, color='r', linestyle='--', linewidth=2,
                        label='Example NDC Target (-30% from BAU)')

            # Plot BAU and scenarios
            plt.plot(years, results['bau']['co2'], 'k-', linewidth=2, label='Business As Usual')

            for i, (scenario, data) in enumerate(results['scenarios'].items()):
                plt.plot(years, data['co2'], marker='o', linestyle='-', linewidth=2,
                         color=colors[i], label=f"{scenario.capitalize()} Efficiency Scenario")

            # Add NDC contribution annotations
            for i, (scenario, data) in enumerate(results['scenarios'].items()):
                scenario_2030_co2 = data['co2'][bau_2030_index]
                reduction = (bau_2030_co2 - scenario_2030_co2) / bau_2030_co2 * 100
                ndc_contribution = (bau_2030_co2 - scenario_2030_co2) / (bau_2030_co2 - ndc_target) * 100

                plt.annotate(f"{scenario.capitalize()}: {reduction:.1f}% reduction\n({ndc_contribution:.1f}% of NDC target)",
                             xy=(2030, scenario_2030_co2), xytext=(2031, scenario_2030_co2),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colors[i]),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8),
                             fontsize=10)

        plt.title(f"{country} - Contribution to NDC Targets", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ndc_contribution.png'), dpi=300)
        plt.close()

    def _generate_report(self, country, forecasts, results, benefits, carbon_credits):
        """
        Generate a summary report for the energy efficiency analysis

        Parameters:
        -----------
        country : str
            Country name
        forecasts : DataFrame
            Combined forecast data
        results : dict
            Scenario results
        benefits : dict
            Benefits calculations
        carbon_credits : dict
            Carbon credit calculations
        """
        print(f"  Generating summary report for {country}")

        output_dir = os.path.join('output', country, 'efficiency_analysis')
        report_file = os.path.join(output_dir, 'energy_efficiency_report.md')

        with open(report_file, 'w') as f:
            f.write(f"# Energy Efficiency Analysis for {country}\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n")

            f.write("## Executive Summary\n\n")

            # Summary of key findings
            f.write("### Key Findings\n\n")

            for scenario in benefits['economic']:
                roi = benefits['economic'][scenario]['roi'] * 100
                payback = benefits['economic'][scenario]['payback_period']
                total_benefit = np.sum(benefits['economic'][scenario]['net_benefit']) / 1e9

                total_co2 = np.sum(benefits['environmental'][scenario]['total_co2_reduction']) / 1e6
                carbon_value = np.sum(carbon_credits[scenario]['annual_value']) / 1e6

                f.write(f"**{scenario.capitalize()} Efficiency Scenario:**\n\n")
                f.write(f"- Return on Investment: {roi:.1f}%\n")
                f.write(f"- Payback Period: {payback:.1f} years\n" if payback else "- Payback Period: Beyond analysis timeframe\n")
                f.write(f"- Total Economic Benefit: ${total_benefit:.2f} billion\n")
                f.write(f"- Total CO2 Reduction: {total_co2:.2f} million metric tons\n")
                f.write(f"- Potential Carbon Credit Value: ${carbon_value:.2f} million\n\n")

            # Recommendations
            f.write("### Recommendations\n\n")

            # Find best scenario based on ROI
            rois = {s: benefits['economic'][s]['roi'] for s in benefits['economic']}
            best_scenario = max(rois, key=rois.get)

            f.write(f"1. **Investment Strategy:** The {best_scenario.capitalize()} efficiency scenario offers the best return on investment.\n")
            f.write("2. **Carbon Credit Aggregation:** Creating a national energy efficiency program that aggregates smaller projects can increase carbon credit value by approximately 20%.\n")
            f.write("3. **NDC Enhancement:** Energy efficiency improvements can significantly contribute to meeting and exceeding NDC targets.\n")
            f.write("4. **Climate Resilience:** Implementing energy efficiency measures will reduce vulnerability to rising energy costs due to climate change.\n\n")

            # Detailed analysis
            f.write("## Detailed Analysis\n\n")

            f.write("### Climate Change Impacts\n\n")

            f.write("Without energy efficiency improvements, climate change is projected to increase energy demand due to:\n\n")
            f.write(f"- Rising temperatures (projected {self.climate_factors['temperature_increase']*100:.1f}% increase per year)\n")
            f.write(f"- Increased cooling demand ({self.climate_factors['cooling_demand_increase']*100:.1f}% increase per year)\n")
            f.write(f"- Economic costs from extreme weather events (estimated {self.climate_factors['extreme_weather_cost']*100:.1f}% of GDP per year)\n\n")

            f.write("### Energy Efficiency Scenarios\n\n")

            for scenario, params in self.efficiency_scenarios.items():
                f.write(f"**{scenario.capitalize()} Scenario:**\n\n")
                f.write(f"- Energy Demand Reduction: {params['demand_reduction']*100:.1f}%\n")
                f.write(f"- CO2 Emission Reduction: {params['co2_reduction']*100:.1f}%\n")
                f.write(f"- Implementation Cost: {params['implementation_cost']*100:.1f}% of GDP\n\n")

            f.write("### Carbon Credit Opportunities\n\n")

            f.write("Energy efficiency projects provide excellent opportunities for carbon credit generation:\n\n")

            f.write("- **Pricing Trajectory:**\n")
            f.write(f"  - Current price: ${self.carbon_credit_pricing['current']} per ton CO2\n")
            f.write(f"  - 2030 projected price: ${self.carbon_credit_pricing['projected_2030']} per ton CO2\n")
            f.write(f"  - 2035 projected price: ${self.carbon_credit_pricing['projected_2035']} per ton CO2\n\n")

            f.write("- **Aggregation Benefits:**\n")
            f.write("  - Small-scale efficiency projects can be difficult to register individually\n")
            f.write("  - Aggregating projects at national or sectoral level increases viability\n")
            f.write("  - Aggregated projects typically receive 15-20% price premiums\n")
            f.write("  - Reduces transaction costs and monitoring requirements\n\n")

            f.write("### NDC Implications\n\n")

            f.write("Energy efficiency improvements can significantly contribute to Nationally Determined Contributions (NDCs):\n\n")

            # Find 2030 values if available
            years = results['years']
            bau_2030_index = np.where(years == 2030)[0][0] if 2030 in years else -1

            if bau_2030_index >= 0:
                bau_2030_co2 = results['bau']['co2'][bau_2030_index]

                for scenario, data in results['scenarios'].items():
                    scenario_2030_co2 = data['co2'][bau_2030_index]
                    reduction = (bau_2030_co2 - scenario_2030_co2) / bau_2030_co2 * 100

                    f.write(f"- **{scenario.capitalize()} Scenario:** {reduction:.1f}% reduction from BAU by 2030\n")

                f.write("\nThese reductions can be included in updated NDC submissions to strengthen national climate commitments.\n\n")

            f.write("## Conclusion\n\n")

            f.write("Energy efficiency investments represent a win-win opportunity for economic development and climate action. By implementing the recommended efficiency measures, " +
                    f"{country} can achieve significant energy cost savings, generate valuable carbon credits, and make substantial progress toward meeting its climate commitments.\n\n")

            f.write("The analysis shows that even accounting for implementation costs, energy efficiency improvements provide positive returns on investment " +
                    "while building resilience against climate change impacts and rising energy costs.\n\n")

            f.write("*Note: This analysis is based on forecasted data and should be revised with more detailed country-specific information for implementation planning.*\n")

        print(f"  Report saved to {report_file}")

    def analyze_all_countries(self):
        """Analyze all available countries"""
        if self.forecaster.selected_country:
            countries = [self.forecaster.selected_country]
        else:
            countries = self.data['Country'].unique()

        results = {}
        for country in countries:
            country_str = str(country)
            results[country_str] = self.analyze_country(country_str)

        return results

def run_efficiency_analysis(base_forecaster):
    """
    Run energy efficiency analysis using base forecaster data

    Parameters:
    -----------
    base_forecaster : ASEANForecaster
        The base forecaster object with data and forecasts
    """
    print("\nStarting Energy Efficiency Analysis")

    analyzer = EnergyEfficiencyAnalyzer(base_forecaster)

    if base_forecaster.selected_country:
        results = analyzer.analyze_country(base_forecaster.selected_country)
    else:
        # Ask which country to analyze
        available_countries = base_forecaster.data['Country'].unique()
        available_countries = [str(country) for country in available_countries]

        print("\nAvailable countries for efficiency analysis:")
        for i, country in enumerate(available_countries, 1):
            print(f"{i}. {country}")

        while True:
            try:
                selection = input("\nSelect a country for efficiency analysis (enter number or name) or 'all' for all countries: ")

                if selection.lower() == 'all':
                    results = analyzer.analyze_all_countries()
                    break

                if selection.isdigit():
                    index = int(selection) - 1
                    if 0 <= index < len(available_countries):
                        country = available_countries[index]
                        results = analyzer.analyze_country(country)
                        break
                    else:
                        print(f"Invalid number. Please enter a number between 1 and {len(available_countries)}")
                elif selection in available_countries:
                    results = analyzer.analyze_country(selection)
                    break
                else:
                    print("Invalid selection. Please try again.")
            except Exception as e:
                print(f"Error: {e}")

    print("\nEnergy Efficiency Analysis Complete!")
    print("Check the 'efficiency_analysis' folder for each country for detailed reports and visualizations.")

    return results

# Standalone mode for direct execution
if __name__ == "__main__":
    import sys

    print("ASEAN Energy Efficiency Analyzer")
    print("--------------------------------")
    print("Note: This file is designed to be imported by the main ASEAN forecaster script.")
    print("      For standalone use, provide a CSV file with forecast data.")
    print()

    # Check if a file path was provided
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Attempting to analyze file: {csv_file}")

        try:
            # Create a minimal forecaster-like object to pass to the analyzer
            class MinimalForecaster:
                def __init__(self, filepath):
                    import pandas as pd
                    self.data = pd.read_csv(filepath)
                    self.selected_country = None

                    # If the data has a Country column, get the first country
                    if 'Country' in self.data.columns:
                        countries = self.data['Country'].unique()
                        if len(countries) > 0:
                            self.selected_country = str(countries[0])
                            print(f"Found country in data: {self.selected_country}")

            # Create a minimal forecaster
            mini_forecaster = MinimalForecaster(csv_file)

            # Run the analysis
            run_efficiency_analysis(mini_forecaster)

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
    else:
        print("Usage examples:")
        print("  python energy_efficiency_analyzer.py path/to/forecast_data.csv")
        print()
        print("Or preferably, use the main script that integrates everything:")
        print("  python run_analysis.py --skip-forecast")

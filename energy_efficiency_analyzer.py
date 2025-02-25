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
    key_indicators = ['Demand (TWh)', 'GDP (current US$)', 'CO2 emissions (metric tons per capita)',
                      'Population, total']
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


class TVerMarketInfo:
    """
    Contains information about Thailand's Voluntary Emission Reduction (T-VER) market.
    """

    def __init__(self):
        # Base T-VER market values from Carbon Pulse article
        self.prices = {
            'agriculture': 30.0,  # ~1,090 baht per tonne CO2 ($30 USD)
            'solar': 3.5,  # ~40-250 baht range (avg ~$3.50 USD)
            'average': 5.0,  # ~174 baht per tonne ($5 USD)
            'industrial': 4.5,  # Estimated based on market data
            'building': 4.8  # Estimated based on market data
        }

        # Premium for aggregated projects
        self.aggregation_premium = 0.2  # 20% premium for aggregated projects

        # Annual price growth based on recent market data
        self.annual_price_growth = 0.08  # 8% annual growth based on 40% increase in one quarter

        # TGO registration and verification costs (% of project value)
        self.registration_cost_percent = 0.03  # 3% for registration and admin
        self.verification_cost_percent = 0.02  # 2% for verification

        # Recent market growth
        self.recent_quarterly_growth = 0.40  # 40% growth in Q1 2024 from previous quarter


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
                'co2_reduction': 0.15,  # 15% reduction in CO2 emissions
                'implementation_cost': 0.02  # 2% of GDP
            },
            'medium': {
                'demand_reduction': 0.20,  # 20% reduction in energy demand
                'co2_reduction': 0.30,  # 30% reduction in CO2 emissions
                'implementation_cost': 0.04  # 4% of GDP
            },
            'high': {
                'demand_reduction': 0.35,  # 35% reduction in energy demand
                'co2_reduction': 0.50,  # 50% reduction in CO2 emissions
                'implementation_cost': 0.07  # 7% of GDP
            }
        }

        # Climate change impact factors (conservative estimates)
        self.climate_factors = {
            'cooling_demand_increase': 0.015,  # 1.5% increase per year in cooling demand
            'temperature_increase': 0.02,  # 2% increase in average temperature per year
            'extreme_weather_cost': 0.005  # 0.5% of GDP impact from extreme weather
        }

        # Add T-VER market information for Thailand
        self.tver_market = TVerMarketInfo()

        # Generic carbon credit pricing (used for non-Thailand countries)
        self.carbon_credit_pricing = {
            'current': 15,  # Current average price
            'projected_2030': 40,  # Projected 2030 price
            'projected_2035': 75  # Projected 2035 price
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

    #-----------------------------------------------------------------------------------------------------------------------
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
                base_indicator,  # Then try direct column
                f"{base_indicator}_Source"  # Then try source indicator
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

            # Initialize results dictionary - UPDATED to remove "without climate change" baseline
            results = {
                'years': years,
                'bau': {  # Business as usual with climate impacts
                    'demand': bau_demand.values,
                    'co2': climate_affected_co2,
                    'gdp': baseline_values['gdp'].values,
                    'population': baseline_values['population'].values
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
                print(f"    Demand reduction: {demand_reduction * 100:.1f}%")
                print(f"    CO2 reduction: {co2_reduction * 100:.1f}%")
                print(f"    Implementation cost: {implementation_cost * 100:.1f}% of GDP")

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
                co2_reduction = results['bau']['co2'] - data['co2']
                total_co2_reduction = co2_reduction * results['bau']['population']
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
                print(f"    Total energy cost savings: ${np.sum(energy_cost_savings) / 1e9:.2f} billion")
                print(f"    Total CO2 reduction: {np.sum(total_co2_reduction) / 1e6:.2f} million tons")
                print(f"    ROI: {roi * 100:.1f}%")
                print(
                    f"    Payback period: {payback_period:.1f} years" if payback_period else "    Payback period: Beyond timeframe")

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
                                        'energy_savings_twh': np.zeros(len(years)),
                                        'energy_cost_savings': np.zeros(len(years)),
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

    def _calculate_carbon_credits(self, country, forecasts, results):
        """
        Calculate potential carbon credits from efficiency improvements with Thailand T-VER market integration

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

        # Determine if we're analyzing Thailand
        is_thailand = country.lower() == 'thailand'

        if is_thailand:
            print("  Using Thailand T-VER market data for carbon credit calculations")
            # Create sector-specific carbon credit projections
            sectors = {
                'agriculture': {
                    'share': 0.20,  # 20% of credits from agriculture/forestry sector
                    'price': self.tver_market.prices['agriculture'],
                    'growth_rate': self.tver_market.annual_price_growth
                },
                'solar': {
                    'share': 0.30,  # 30% of credits from solar/renewable sector
                    'price': self.tver_market.prices['solar'],
                    'growth_rate': self.tver_market.annual_price_growth * 0.8  # Lower growth for renewables
                },
                'industrial': {
                    'share': 0.35,  # 35% of credits from industrial efficiency
                    'price': self.tver_market.prices['industrial'],
                    'growth_rate': self.tver_market.annual_price_growth * 1.1  # Higher growth for industry
                },
                'building': {
                    'share': 0.15,  # 15% of credits from buildings sector
                    'price': self.tver_market.prices['building'],
                    'growth_rate': self.tver_market.annual_price_growth * 0.9
                }
            }

            # Create weighted average carbon price for each year
            carbon_prices = []
            sector_prices = []

            for year_idx, year in enumerate(years):
                year_prices = {}
                years_from_now = year_idx  # Index starts at 0 for first forecast year

                weighted_price = 0
                for sector, info in sectors.items():
                    # Calculate compounding price growth for this sector
                    sector_price = info['price'] * (1 + info['growth_rate']) ** years_from_now
                    weighted_price += sector_price * info['share']
                    year_prices[sector] = sector_price

                carbon_prices.append(weighted_price)
                sector_prices.append(year_prices)

            print(f"  T-VER price trajectory: ${carbon_prices[0]:.2f} (2024) to ${carbon_prices[-1]:.2f} (2035)")
        else:
            # Use the standard carbon price trajectory for non-Thailand countries
            start_price = self.carbon_credit_pricing['current']
            end_price = self.carbon_credit_pricing['projected_2035']

            # Linear interpolation for prices
            carbon_prices = []
            for year in years:
                if year <= 2030:
                    # Linear increase from current to 2030 price
                    progress = (year - min(years)) / (2030 - min(years))
                    price = start_price + progress * (self.carbon_credit_pricing['projected_2030'] - start_price)
                else:
                    # Linear increase from 2030 to 2035 price
                    progress = (year - 2030) / (2035 - 2030)
                    price = self.carbon_credit_pricing['projected_2030'] + progress * (
                                end_price - self.carbon_credit_pricing['projected_2030'])
                carbon_prices.append(price)

        # Calculate credits for each scenario
        for scenario, data in results['scenarios'].items():
            co2_reduction = results['bau']['co2'] - data['co2']
            total_co2_reduction = co2_reduction * results['bau']['population']

            # Calculate carbon credit value
            carbon_credit_value = np.array([total_co2_reduction[i] * carbon_prices[i]
                                            for i in range(len(years))])

            # Calculate cumulative credit generation
            cumulative_credits = np.cumsum(total_co2_reduction)
            cumulative_value = np.cumsum(carbon_credit_value)

            # Calculate aggregation potential (combining small projects into larger portfolios)
            aggregation_premium = self.tver_market.aggregation_premium if is_thailand else 0.2
            aggregation_potential = carbon_credit_value * (1 + aggregation_premium)

            # Calculate sector breakdown for Thailand
            sector_breakdown = None
            if is_thailand:
                sector_breakdown = []
                for i in range(len(years)):
                    year_breakdown = {}
                    for sector, info in sectors.items():
                        # Calculate credits by sector
                        sector_credits = total_co2_reduction[i] * info['share']
                        sector_value = sector_credits * sector_prices[i][sector]
                        year_breakdown[sector] = {
                            'credits': sector_credits,
                            'value': sector_value,
                            'price': sector_prices[i][sector]
                        }
                    sector_breakdown.append(year_breakdown)

            # Calculate TGO registration and verification costs for Thailand
            tgo_costs = None
            if is_thailand:
                registration_cost = carbon_credit_value * self.tver_market.registration_cost_percent
                verification_cost = carbon_credit_value * self.tver_market.verification_cost_percent
                total_tgo_cost = registration_cost + verification_cost
                tgo_costs = {
                    'registration': registration_cost,
                    'verification': verification_cost,
                    'total': total_tgo_cost
                }

            carbon_credits[scenario] = {
                'annual_credits': total_co2_reduction,
                'annual_value': carbon_credit_value,
                'cumulative_credits': cumulative_credits,
                'cumulative_value': cumulative_value,
                'aggregation_potential': aggregation_potential,
                'carbon_prices': carbon_prices
            }

            # Add Thailand-specific information
            if is_thailand:
                carbon_credits[scenario].update({
                    'sector_breakdown': sector_breakdown,
                    'tgo_costs': tgo_costs,
                    'is_tver': True
                })

        return carbon_credits


#---------------------------------------------------------------------------------------------------------------------------------
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

        # Check if this is Thailand
        is_thailand = country.lower() == 'thailand'

        output_dir = os.path.join('output', country, 'efficiency_analysis', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        years = results['years']

        # 1. Energy Demand Scenarios - UPDATED to remove "without climate change" baseline
        plt.figure(figsize=(12, 8))
        plt.plot(years, results['bau']['demand'], 'r-', linewidth=3, label='Business As Usual')

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

        # 2. CO2 Emissions Scenarios - UPDATED to remove "without climate change" baseline
        plt.figure(figsize=(12, 8))
        plt.plot(years, results['bau']['co2'], 'r-', linewidth=3, label='Business As Usual')

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
            # For Thailand, include carbon credit revenue in the economic benefits
            if is_thailand:
                carbon_revenue = carbon_credits[scenario]['annual_value'] / 1e9  # Convert to billions
                energy_savings = data['energy_cost_savings'] / 1e9
                env_damage = benefits['environmental'][scenario]['env_damage_avoided'] / 1e9
                implementation = data['implementation_cost'] / 1e9

                # Stacked bar with energy savings and carbon revenue
                plt.bar(x + i * bar_width, energy_savings, width=bar_width,
                        label=f"{scenario.capitalize()} Energy Savings", color=colors[i])
                plt.bar(x + i * bar_width, carbon_revenue, width=bar_width, bottom=energy_savings,
                        label=f"{scenario.capitalize()} Carbon Credits", color=colors[i], alpha=0.6)
            else:
                # Original plot for non-Thailand countries
                plt.bar(x + i * bar_width, data['net_benefit'] / 1e9, width=bar_width,
                        label=f"{scenario.capitalize()} Scenario", alpha=0.7)

        title = f"{country} - Economic Benefits of Energy Efficiency"
        if is_thailand:
            title += " (Including T-VER Revenue)"

        plt.title(title, fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Billion USD", fontsize=12)
        plt.xticks(x + bar_width, years)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'economic_benefits.png'), dpi=300)
        plt.close()

        # 4. Carbon Credits Value
        plt.figure(figsize=(12, 8))

        for i, (scenario, data) in enumerate(carbon_credits.items()):
            plt.plot(years, data['annual_value'] / 1e6, marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Scenario")

            # For Thailand, also show the aggregation potential
            if is_thailand:
                plt.plot(years, data['aggregation_potential'] / 1e6, marker='x', linestyle='--', linewidth=1,
                         color=colors[i], label=f"{scenario.capitalize()} with Aggregation Premium")

        title = f"{country} - Annual Carbon Credit Value"
        if is_thailand:
            title += " (T-VER Market)"

        plt.title(title, fontsize=15)
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

                plt.annotate(
                    f"{scenario.capitalize()}: {reduction:.1f}% reduction\n({ndc_contribution:.1f}% of NDC target)",
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


    def _create_thailand_specific_visualizations(self, country, results, benefits, carbon_credits):
        """
        Create Thailand-specific visualizations for T-VER market analysis

        Parameters:
        -----------
        country : str
            Country name
        results : dict
            Scenario results
        benefits : dict
            Benefits calculations
        carbon_credits : dict
            Carbon credit calculations
        """
        if country.lower() != 'thailand':
            return

        print(f"  Creating Thailand T-VER market visualizations")

        output_dir = os.path.join('output', country, 'efficiency_analysis', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        years = results['years']

        # 1. T-VER Sector Value Comparison
        plt.figure(figsize=(12, 8))

        # Use medium scenario as an example
        scenario = 'medium'
        if scenario in carbon_credits:
            sector_data = carbon_credits[scenario]['sector_breakdown']

            # Extract data for each sector across years
            sectors = ['agriculture', 'solar', 'industrial', 'building']
            sector_values = {sector: [] for sector in sectors}

            for year_data in sector_data:
                for sector in sectors:
                    sector_values[sector].append(year_data[sector]['value'] / 1e6)  # Convert to millions

            # Create stacked bar chart
            bottom = np.zeros(len(years))
            colors = ['#8fbc8f', '#f4a460', '#778899', '#4682b4']

            for i, sector in enumerate(sectors):
                plt.bar(years, sector_values[sector], bottom=bottom, label=f"{sector.capitalize()}", color=colors[i])
                bottom += np.array(sector_values[sector])

            plt.title(f"Thailand - T-VER Credit Value by Sector ({scenario.capitalize()} Scenario)", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Carbon Credit Value (Million USD)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tver_sector_value.png'), dpi=300)
            plt.close()

        # 2. T-VER Price Trajectory
        plt.figure(figsize=(12, 6))

        # Get price data for different sectors from the first scenario
        first_scenario = list(carbon_credits.keys())[0]
        if 'sector_breakdown' in carbon_credits[first_scenario]:
            sector_data = carbon_credits[first_scenario]['sector_breakdown']
            sectors = ['agriculture', 'solar', 'industrial', 'building', 'average']

            # Extract price trajectories
            price_data = {sector: [] for sector in sectors}

            # Get sector prices
            for year_idx, year_data in enumerate(sector_data):
                for sector in sectors:
                    if sector == 'average':
                        price_data[sector].append(carbon_credits[first_scenario]['carbon_prices'][year_idx])
                    else:
                        price_data[sector].append(year_data[sector]['price'])

            # Plot price trajectories
            colors = ['#8fbc8f', '#f4a460', '#778899', '#4682b4', '#000000']
            linestyles = ['-', '--', '-.', ':', '-']

            for i, sector in enumerate(sectors):
                plt.plot(years, price_data[sector], label=f"{sector.capitalize()}",
                         color=colors[i], linestyle=linestyles[i], linewidth=2)

            plt.title(f"Thailand - T-VER Price Projections by Sector", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Carbon Price (USD/tonne CO2)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tver_price_projections.png'), dpi=300)
            plt.close()

        # 3. Cost comparison - TGO costs vs revenue
        plt.figure(figsize=(12, 6))

        if scenario in carbon_credits and 'tgo_costs' in carbon_credits[scenario]:
            tgo_costs = carbon_credits[scenario]['tgo_costs']

            # Convert to millions
            registration_costs = tgo_costs['registration'] / 1e6
            verification_costs = tgo_costs['verification'] / 1e6
            credit_revenue = carbon_credits[scenario]['annual_value'] / 1e6

            # Create stacked bar for costs and a line for revenue
            plt.bar(years, registration_costs, label="TGO Registration Costs", color='#CD5C5C', alpha=0.7)
            plt.bar(years, verification_costs, bottom=registration_costs, label="TGO Verification Costs", color='#F08080',
                    alpha=0.7)
            plt.plot(years, credit_revenue, 'k-', linewidth=2, label="Credit Revenue")

            plt.title(f"Thailand - T-VER Costs vs Revenue ({scenario.capitalize()} Scenario)", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Million USD", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tver_costs_vs_revenue.png'), dpi=300)
            plt.close()


    def _calculate_tver_loan_repayment(self, country, results, benefits, carbon_credits, loan_amount, world_bank_share=0.4):
        """
        Calculate loan repayment schedule incorporating T-VER revenue

        Parameters:
        -----------
        country : str
            Country name
        results : dict
            Scenario results
        benefits : dict
            Benefits calculations
        carbon_credits : dict
            Carbon credit calculations
        loan_amount : float
            Initial loan amount in USD
        world_bank_share : float
            Share of savings allocated to loan repayment (default: 0.4)

        Returns:
        --------
        dict
            Loan repayment analysis
        """
        if country.lower() != 'thailand':
            return None

        print(f"  Calculating T-VER enhanced loan repayment for Thailand")

        years = results['years']
        repayment_analysis = {}

        # Calculate for each scenario
        for scenario in benefits['economic']:
            # Get energy cost savings
            energy_savings = benefits['economic'][scenario]['energy_cost_savings']

            # Get carbon credit revenue
            carbon_revenue = carbon_credits[scenario]['annual_value']

            # Calculate payment streams
            energy_payment = energy_savings * world_bank_share
            carbon_payment = carbon_revenue * 0.8  # Assume 80% of carbon credits go to loan repayment

            # Calculate total annual payment
            total_payment = energy_payment + carbon_payment

            # Calculate remaining balance
            remaining_balance = np.zeros(len(years))
            remaining_balance[0] = loan_amount - total_payment[0]
            for i in range(1, len(years)):
                if remaining_balance[i - 1] > 0:
                    remaining_balance[i] = remaining_balance[i - 1] - total_payment[i]
                else:
                    remaining_balance[i] = 0

            # Find payback period
            payback_period = None
            payback_year = None
            for i, year in enumerate(years):
                if remaining_balance[i] <= 0:
                    if i > 0:
                        # Interpolate for more precise estimate
                        prev_balance = remaining_balance[i - 1]
                        current_payment = total_payment[i]
                        fraction = prev_balance / current_payment
                        payback_period = (i - 1) + fraction
                        payback_year = years[i - 1] + fraction
                    else:
                        payback_period = 0
                        payback_year = years[0]
                    break

            # Store results
            repayment_analysis[scenario] = {
                'years': years,
                'energy_payment': energy_payment,
                'carbon_payment': carbon_payment,
                'total_payment': total_payment,
                'remaining_balance': remaining_balance,
                'payback_period': payback_period,
                'payback_year': payback_year
            }

            # Print summary
            if payback_period is not None:
                print(
                    f"    {scenario.capitalize()} scenario payback period: {payback_period:.2f} years (by {payback_year:.2f})")
                print(f"    Energy payments: ${np.sum(energy_payment) / 1e9:.2f} billion")
                print(f"    Carbon payments: ${np.sum(carbon_payment) / 1e9:.2f} billion")
            else:
                print(f"    {scenario.capitalize()} scenario does not reach payback within timeframe")

        return repayment_analysis


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
                f.write(
                    f"- Payback Period: {payback:.1f} years\n" if payback else "- Payback Period: Beyond analysis timeframe\n")
                f.write(f"- Total Economic Benefit: ${total_benefit:.2f} billion\n")
                f.write(f"- Total CO2 Reduction: {total_co2:.2f} million metric tons\n")
                f.write(f"- Potential Carbon Credit Value: ${carbon_value:.2f} million\n\n")

            # Recommendations
            f.write("### Recommendations\n\n")

            # Find best scenario based on ROI
            rois = {s: benefits['economic'][s]['roi'] for s in benefits['economic']}
            best_scenario = max(rois, key=rois.get)

            f.write(
                f"1. **Investment Strategy:** The {best_scenario.capitalize()} efficiency scenario offers the best return on investment.\n")
            f.write(
                "2. **Carbon Credit Aggregation:** Creating a national energy efficiency program that aggregates smaller projects can increase carbon credit value by approximately 20%.\n")
            f.write(
                "3. **NDC Enhancement:** Energy efficiency improvements can significantly contribute to meeting and exceeding NDC targets.\n")
            f.write(
                "4. **Climate Resilience:** Implementing energy efficiency measures will reduce vulnerability to rising energy costs due to climate change.\n\n")

            # Detailed analysis
            f.write("## Detailed Analysis\n\n")

            f.write("### Climate Change Impacts\n\n")

            f.write(
                "All scenarios incorporate climate change impacts, which are projected to increase energy demand due to:\n\n")
            f.write(
                f"- Rising temperatures (projected {self.climate_factors['temperature_increase'] * 100:.1f}% increase per year)\n")
            f.write(
                f"- Increased cooling demand ({self.climate_factors['cooling_demand_increase'] * 100:.1f}% increase per year)\n")
            f.write(
                f"- Economic costs from extreme weather events (estimated {self.climate_factors['extreme_weather_cost'] * 100:.1f}% of GDP per year)\n\n")

            f.write("### Energy Efficiency Scenarios\n\n")

            for scenario, params in self.efficiency_scenarios.items():
                f.write(f"**{scenario.capitalize()} Scenario:**\n\n")
                f.write(f"- Energy Demand Reduction: {params['demand_reduction'] * 100:.1f}%\n")
                f.write(f"- CO2 Emission Reduction: {params['co2_reduction'] * 100:.1f}%\n")
                f.write(f"- Implementation Cost: {params['implementation_cost'] * 100:.1f}% of GDP\n\n")

            f.write("### Carbon Credit Opportunities\n\n")

            # Thailand-specific carbon pricing info
            if country.lower() == 'thailand':
                f.write(
                    "Thailand's Voluntary Emission Reduction (T-VER) program offers excellent opportunities for carbon credit generation:\n\n")
                f.write("- **Current T-VER Prices by Sector:**\n")
                f.write(f"  - Agriculture/Forestry: ${self.tver_market.prices['agriculture']} per ton CO2\n")
                f.write(f"  - Solar Energy: ${self.tver_market.prices['solar']} per ton CO2\n")
                f.write(f"  - Industrial: ${self.tver_market.prices['industrial']} per ton CO2\n")
                f.write(f"  - Building: ${self.tver_market.prices['building']} per ton CO2\n")
                f.write(f"  - Market Average: ${self.tver_market.prices['average']} per ton CO2\n\n")
            else:
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

            f.write(
                "Energy efficiency improvements can significantly contribute to Nationally Determined Contributions (NDCs):\n\n")

            # Find 2030 values if available
            years = results['years']
            bau_2030_index = np.where(years == 2030)[0][0] if 2030 in years else -1

            if bau_2030_index >= 0:
                bau_2030_co2 = results['bau']['co2'][bau_2030_index]

                for scenario, data in results['scenarios'].items():
                    scenario_2030_co2 = data['co2'][bau_2030_index]
                    reduction = (bau_2030_co2 - scenario_2030_co2) / bau_2030_co2 * 100

                    f.write(f"- **{scenario.capitalize()} Scenario:** {reduction:.1f}% reduction from BAU by 2030\n")

                f.write(
                    "\nThese reductions can be included in updated NDC submissions to strengthen national climate commitments.\n\n")

            f.write("## Conclusion\n\n")

            f.write(
                "Energy efficiency investments represent a win-win opportunity for economic development and climate action. By implementing the recommended efficiency measures, " +
                f"{country} can achieve significant energy cost savings, generate valuable carbon credits, and make substantial progress toward meeting its climate commitments.\n\n")

            f.write(
                "The analysis shows that even accounting for implementation costs, energy efficiency improvements provide positive returns on investment " +
                "while building resilience against climate change impacts and rising energy costs.\n\n")

            f.write(
                "*Note: This analysis is based on forecasted data and should be revised with more detailed country-specific information for implementation planning.*\n")

        print(f"  Report saved to {report_file}")


    def _generate_tver_integration_report(self, country, forecasts, results, benefits, carbon_credits, loan_repayment=None):
        """
        Generate a T-VER integration report for Thailand

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
        loan_repayment : dict, optional
            Loan repayment analysis
        """
        if country.lower() != 'thailand':
            return

        print(f"  Generating T-VER integration report for Thailand")

        output_dir = os.path.join('output', country, 'efficiency_analysis')
        report_file = os.path.join(output_dir, 'tver_integration_report.md')

        with open(report_file, 'w') as f:
            f.write(f"# Thailand Voluntary Emission Reduction (T-VER) Integration Report\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n")

            f.write("## Executive Summary\n\n")
            f.write(
                "This report analyzes the integration of Thailand's Voluntary Emission Reduction Programme (T-VER) with energy efficiency initiatives, highlighting the potential carbon credit revenue streams, project types, and economic benefits.\n\n")

            # Summary of key findings
            scenario = 'medium'  # Use medium scenario for summary
            if scenario in carbon_credits and scenario in benefits['economic']:
                total_co2 = np.sum(carbon_credits[scenario]['annual_credits']) / 1e6
                carbon_value = np.sum(carbon_credits[scenario]['annual_value']) / 1e6
                energy_savings = np.sum(benefits['economic'][scenario]['energy_savings_twh'])

                f.write("### Key Findings\n\n")
                f.write(f"- **Total CO2 Reduction Potential:** {total_co2:.2f} million tons over 12 years\n")
                f.write(f"- **Carbon Credit Value:** ${carbon_value:.2f} million using T-VER mechanism\n")
                f.write(f"- **Energy Savings:** {energy_savings:.2f} TWh\n")

                if loan_repayment and scenario in loan_repayment:
                    repayment = loan_repayment[scenario]
                    if repayment['payback_period'] is not None:
                        f.write(
                            f"- **Loan Repayment Period:** {repayment['payback_period']:.2f} years with T-VER revenue\n")
                        energy_contribution = np.sum(repayment['energy_payment']) / (
                                    np.sum(repayment['energy_payment']) + np.sum(repayment['carbon_payment'])) * 100
                        carbon_contribution = np.sum(repayment['carbon_payment']) / (
                                    np.sum(repayment['energy_payment']) + np.sum(repayment['carbon_payment'])) * 100
                        f.write(
                            f"- **Repayment Sources:** {energy_contribution:.1f}% from energy savings, {carbon_contribution:.1f}% from carbon credits\n")

                f.write("\n")

            # T-VER Market Overview
            f.write("## T-VER Market Overview\n\n")
            f.write(
                "The Thailand Voluntary Emission Reduction Programme (T-VER) is managed by the Thailand Greenhouse Gas Management Organization (TGO). The market has shown significant growth, with a 40% price increase in Q1 2024 compared to the previous quarter.\n\n")

            f.write("### Current Carbon Prices by Sector\n\n")
            f.write("| Sector | Price (USD/tCO2) | Price (Thai Baht/tCO2) |\n")
            f.write("|--------|------------------|------------------------|\n")
            f.write(f"| Agriculture/Forestry | ${self.tver_market.prices['agriculture']:.2f} | ~1,090 |\n")
            f.write(f"| Solar Energy | ${self.tver_market.prices['solar']:.2f} | 40-250 |\n")
            f.write(f"| Industrial | ${self.tver_market.prices['industrial']:.2f} | ~160 |\n")
            f.write(f"| Building | ${self.tver_market.prices['building']:.2f} | ~170 |\n")
            f.write(f"| Market Average | ${self.tver_market.prices['average']:.2f} | ~174 |\n\n")

            # Sector Analysis
            f.write("## Sector Analysis\n\n")

            if scenario in carbon_credits and 'sector_breakdown' in carbon_credits[scenario]:
                sector_data = carbon_credits[scenario]['sector_breakdown']
                sectors = ['agriculture', 'solar', 'industrial', 'building']

                f.write("### Carbon Credit Potential by Sector\n\n")
                f.write("| Sector | Credit Volume (MtCO2) | Credit Value (Million USD) | Share of Total Value |\n")
                f.write("|--------|------------------------|----------------------------|---------------------|\n")

                for sector in sectors:
                    # Sum up credits and value across all years
                    total_credits = sum(year_data[sector]['credits'] for year_data in sector_data) / 1e6
                    total_value = sum(year_data[sector]['value'] for year_data in sector_data) / 1e6
                    total_all_sectors = sum(sum(year_data[s]['value'] for s in sectors) for year_data in sector_data) / 1e6
                    share = (total_value / total_all_sectors) * 100

                    f.write(f"| {sector.capitalize()} | {total_credits:.2f} | ${total_value:.2f} | {share:.1f}% |\n")

                f.write("\n")

            # Implementation Strategy
            f.write("## T-VER Implementation Strategy\n\n")

            f.write("### Project Development Process\n\n")
            f.write("1. **Project Design:** Develop energy efficiency measures aligned with T-VER methodologies\n")
            f.write("2. **Registration with TGO:** Register projects under the T-VER programme\n")
            f.write("3. **Implementation:** Execute energy efficiency measures with proper monitoring\n")
            f.write("4. **Verification:** Verify emission reductions through TGO-approved verifiers\n")
            f.write("5. **Credit Issuance:** Receive T-VER credits based on verified reductions\n")
            f.write("6. **Credit Utilization:** Use credits for loan repayment or contribute to NDC targets\n\n")

            f.write("### Priority Sectors\n\n")
            f.write(
                "Based on current T-VER prices and energy efficiency potential, the following sectors should be prioritized:\n\n")
            f.write("1. **Agricultural Sector:** Highest carbon credit prices (~$30/tCO2)\n")
            f.write("   - Irrigation pump efficiency improvements\n")
            f.write("   - Agricultural processing equipment upgrades\n")
            f.write("   - Biomass energy projects\n\n")

            f.write("2. **Industrial Sector:** Large emission reduction potential\n")
            f.write("   - Motor and drive system efficiency\n")
            f.write("   - Process heat recovery\n")
            f.write("   - Industrial HVAC optimization\n\n")

            f.write("3. **Building Sector:** Good urban implementation potential\n")
            f.write("   - Commercial building energy management systems\n")
            f.write("   - Lighting retrofits\n")
            f.write("   - Building envelope improvements\n\n")

            f.write("4. **Solar Energy:** Complements efficiency measures\n")
            f.write("   - Rooftop solar installations\n")
            f.write("   - Solar water heating\n")
            f.write("   - Industrial solar thermal applications\n\n")

            # Loan Repayment Analysis
            if loan_repayment:
                f.write("## Loan Repayment Analysis\n\n")

                f.write(
                    "The integration of T-VER carbon credits can significantly accelerate loan repayment by providing additional revenue streams beyond energy cost savings.\n\n")

                for scenario_name, repayment in loan_repayment.items():
                    f.write(f"### {scenario_name.capitalize()} Scenario\n\n")

                    if repayment['payback_period'] is not None:
                        f.write(f"- **Payback Period:** {repayment['payback_period']:.2f} years\n")
                        f.write(f"- **Payback Completion:** Year {repayment['payback_year']:.1f}\n")

                        total_energy = np.sum(repayment['energy_payment']) / 1e6
                        total_carbon = np.sum(repayment['carbon_payment']) / 1e6
                        total_payment = total_energy + total_carbon

                        f.write(
                            f"- **Total Energy Savings Payment:** ${total_energy:.2f} million ({(total_energy / total_payment * 100):.1f}%)\n")
                        f.write(
                            f"- **Total Carbon Credit Payment:** ${total_carbon:.2f} million ({(total_carbon / total_payment * 100):.1f}%)\n")
                        f.write(f"- **Total Repayment Amount:** ${total_payment:.2f} million\n\n")
                    else:
                        f.write("- No complete payback within the analysis timeframe.\n\n")

                    f.write("#### Annual Repayment Schedule\n\n")
                    f.write(
                        "| Year | Energy Payment ($M) | Carbon Payment ($M) | Total Payment ($M) | Remaining Balance ($M) |\n")
                    f.write(
                        "|------|---------------------|---------------------|--------------------|-----------------------|\n")

                    for i, year in enumerate(repayment['years']):
                        energy_payment = repayment['energy_payment'][i] / 1e6
                        carbon_payment = repayment['carbon_payment'][i] / 1e6
                        total_payment = repayment['total_payment'][i] / 1e6
                        remaining = repayment['remaining_balance'][i] / 1e6

                        f.write(
                            f"| {year} | ${energy_payment:.2f} | ${carbon_payment:.2f} | ${total_payment:.2f} | ${max(0, remaining):.2f} |\n")

                    f.write("\n")

            # NDC Contribution
            f.write("## Contribution to Thailand's NDC\n\n")
            f.write(
                "Thailand's Nationally Determined Contribution (NDC) under the Paris Agreement targets greenhouse gas emissions reduction. Energy efficiency projects registered under T-VER can contribute to these national targets:\n\n")

            # Calculate contribution to a hypothetical 20% reduction target by 2030
            if 'high' in carbon_credits and 'bau' in results and 2030 in results['years']:
                idx_2030 = list(results['years']).index(2030)
                bau_emissions = results['bau']['co2'][idx_2030] * results['bau']['population'][idx_2030]
                reduced_emissions = results['scenarios']['high']['co2'][idx_2030] * results['bau']['population'][idx_2030]
                reduction_percentage = (bau_emissions - reduced_emissions) / bau_emissions * 100

                f.write(
                    f"- High efficiency scenario could deliver ~{reduction_percentage:.1f}% reduction from BAU by 2030\n")
                f.write("- This represents a significant portion of Thailand's NDC commitment\n")
                f.write("- T-VER credits from these reductions can be used for:\n")
                f.write("  - Domestic offsetting under the upcoming carbon tax\n")
                f.write("  - International voluntary market sales\n")
                f.write("  - Direct counting towards NDC targets\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            f.write(
                "1. **Establish T-VER Project Pipeline:** Develop a portfolio of energy efficiency projects aligned with T-VER methodologies\n")
            f.write(
                "2. **Focus on High-Value Sectors:** Prioritize agriculture and industrial efficiency projects with higher credit prices\n")
            f.write(
                "3. **Aggregate Small Projects:** Combine smaller projects to reduce transaction costs and achieve price premiums\n")
            f.write(
                "4. **Integrate with Carbon Tax:** Position energy efficiency projects to benefit from Thailand's planned carbon tax\n")
            f.write(
                "5. **Enhance MRV Systems:** Invest in robust monitoring, reporting, and verification to maximize credit issuance\n")
            f.write(
                "6. **Use Carbon Revenue for Repayment:** Dedicate a portion of carbon credit revenue to accelerate loan repayment\n")
            f.write(
                "7. **Explore Advance Market Commitments:** Secure upfront financing by selling future carbon credits to interested buyers\n\n")

            f.write("## Conclusion\n\n")
            f.write(
                "The integration of Thailand's T-VER programme with energy efficiency initiatives presents a significant opportunity to accelerate the transition to a low-carbon economy while generating financial returns. The higher prices for agricultural and forestry credits make these sectors particularly attractive, while the industrial sector offers large volume potential. By strategically developing projects that align with T-VER methodologies and Thailand's NDC commitments, the country can achieve substantial emission reductions while accelerating the repayment of energy efficiency investments.\n\n")

            f.write(
                "*Note: This analysis is based on forecasted data and current T-VER market conditions. Actual results may vary based on market developments and policy changes.*\n")

        print(f"  T-VER integration report saved to {report_file}")


    def analyze_country(self, country):
        """
        Perform energy efficiency analysis for a specific country with T-VER integration for Thailand

        Parameters:
        -----------
        country : str
            Country name
        """
        print(f"\nAnalyzing energy efficiency potential for {country}")

        # Convert country to string for consistency
        country = str(country)
        is_thailand = country.lower() == 'thailand'

        # Load combined forecast data
        forecast_file = os.path.join('output', country, f'{country}_combined_forecast.csv')

        if not os.path.exists(forecast_file):
            print(f"  Error: No forecast data found for {country}. Run forecasts first.")
            return None

        try:
            forecasts = pd.read_csv(forecast_file)

            # Check if required indicators exist (in any form - original or forecast)
            required_base_indicators = ['Demand (TWh)', 'GDP (current US$)', 'CO2 emissions (metric tons per capita)',
                                        'Population, total']

            # Check for basic existence of indicators (in any form)
            missing_indicators = []
            for indicator in required_base_indicators:
                if indicator not in forecasts.columns and not any(
                        col.startswith(indicator + '_') for col in forecasts.columns):
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

            # For Thailand, calculate T-VER loan repayment
            loan_repayment = None
            if is_thailand:
                # Estimate loan amount based on medium scenario benefits
                scenario = 'medium'
                if scenario in benefits['economic']:
                    energy_savings = benefits['economic'][scenario]['energy_savings_twh']
                    energy_cost_savings = benefits['economic'][scenario]['energy_cost_savings']

                    # Use first 5 years of cost savings as loan amount
                    loan_amount = np.sum(energy_cost_savings[:5])
                    loan_repayment = self._calculate_tver_loan_repayment(
                        country, results, benefits, carbon_credits, loan_amount, world_bank_share=0.4
                    )

            # Generate standard visualizations
            self._create_visualizations(country, forecasts, results, benefits, carbon_credits)

            # For Thailand, generate additional T-VER specific visualizations
            if is_thailand:
                self._create_thailand_specific_visualizations(country, results, benefits, carbon_credits)

            # Generate standard report
            self._generate_report(country, forecasts, results, benefits, carbon_credits)

            # For Thailand, generate T-VER integration report
            if is_thailand:
                self._generate_tver_integration_report(country, forecasts, results, benefits, carbon_credits,
                                                       loan_repayment)

            return results

        except Exception as e:
            print(f"  Error analyzing {country}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


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
                selection = input(
                    "\nSelect a country for efficiency analysis (enter number or name) or 'all' for all countries: ")

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

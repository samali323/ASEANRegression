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


class CarbonMarketInfo:
    """
    Contains information about carbon market pricing and mechanisms
    """

    def __init__(self):
        # Carbon pricing by sector (USD per ton CO2)
        self.prices = {
            'agriculture': 12.0,  # Agricultural projects
            'solar': 8.0,         # Solar PV
            'industrial': 7.5,    # Industrial energy efficiency
            'building': 7.0,      # Building & infrastructure EE
            'average': 8.5        # Average price
        }

        # Premium for aggregated projects
        self.aggregation_premium = 0.15  # 15% premium for aggregation

        # Annual price growth projection
        self.annual_price_growth = 0.06  # 6% annual growth

        # Management and verification costs (% of project value)
        self.management_fee_percent = 0.10  # 10% management fee
        self.registration_cost_percent = 0.02  # 2% for registration
        self.verification_cost_percent = 0.015  # 1.5% for verification


class EnergyEfficiencyAnalyzer:
    """
    Analyzes energy efficiency potential, carbon savings, and economic benefits
    based on forecaster outputs.
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

        # === PROGRAM CONFIGURATION CONSTANTS ===

        # Loan configuration
        self.loan_amount = 4e8         # $400 million loan
        self.loan_interest_rate = 0.02  # 2% annual interest
        self.loan_grace_period = 5      # 5-year grace period

        # Revenue allocation
        self.energy_savings_allocation = 0.4  # 40% of energy savings to loan repayment
        self.carbon_revenue_allocation = 0.8  # 80% of carbon revenue to loan repayment

        # Energy cost factors
        self.energy_cost_per_mwh = 120       # $120 per MWh (0.12/kWh)
        self.energy_price_increase = 0.02    # 2% annual increase

        # Environmental damage costs
        self.carbon_damage_cost = 50         # $50 per ton CO2
        self.carbon_damage_increase = 0.025  # 2.5% annual increase

        # Carbon crediting factor
        self.crediting_factor = 0.60         # 60% of reductions eligible for credits

        # Implementation cost efficiency
        self.implementation_cost_factor = 0.6  # 40% reduction from economies of scale

        # Default efficiency improvement scenarios with realistic values
        self.efficiency_scenarios = efficiency_scenarios or {
            'low': {
                'demand_reduction': 0.12,  # 12% reduction (LED streetlighting, basic EE)
                'co2_reduction': 0.18,     # 18% CO2 reduction
                'implementation_cost': 0.008  # 0.8% of GDP
            },
            'medium': {
                'demand_reduction': 0.25,  # 25% reduction (comprehensive urban EE program)
                'co2_reduction': 0.35,     # 35% CO2 reduction
                'implementation_cost': 0.015  # 1.5% of GDP
            },
            'high': {
                'demand_reduction': 0.40,  # 40% reduction (ambitious program across all sectors)
                'co2_reduction': 0.55,     # 55% CO2 reduction
                'implementation_cost': 0.025  # 2.5% of GDP
            }
        }

        # Climate change impact factors
        self.climate_factors = {
            'cooling_demand_increase': 0.006,  # 0.6% increase per year
            'temperature_increase': 0.003,     # 0.3% increase per year
            'extreme_weather_cost': 0.003      # 0.3% of GDP impact
        }

        # Add carbon market information
        self.carbon_market = CarbonMarketInfo()

        # Generic carbon credit pricing (used for non-Thailand countries)
        self.carbon_credit_pricing = {
            'current': 8,      # Current price
            'projected_2030': 15,  # 2030 price
            'projected_2035': 25   # 2035 price
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

    def validate_forecast_data(self, forecasts, country):
        """
        Validate forecast data before running analysis

        Parameters:
        -----------
        forecasts : DataFrame
            The forecast data to validate
        country : str
            Country name for logging

        Returns:
        --------
        bool
            True if validation passes, False otherwise
        """
        print(f"\nValidating forecast data for {country}")
        validation_passed = True

        # Check for Prophet forecasts
        prophet_columns = [col for col in forecasts.columns if '_Prophet' in col]
        print(f"Found {len(prophet_columns)} Prophet forecast columns: {prophet_columns}")

        if len(prophet_columns) == 0:
            print("WARNING: No Prophet forecast columns found. Analysis may not be accurate.")
            validation_passed = False

        # Check forecast ranges
        years = forecasts['Year'].unique()
        print(f"Years in forecast: {min(years)} to {max(years)}")

        future_years = [y for y in years if y >= 2024]
        if len(future_years) == 0:
            print("ERROR: No future years (2024+) found in forecast data.")
            validation_passed = False

        # Sample key indicators to verify reasonable values
        key_indicators = {
            'Demand (TWh)': {'realistic_min': 50, 'realistic_max': 1000},
            'GDP (current US$)': {'realistic_min': 1e10, 'realistic_max': 1e13},
            'CO2 emissions (metric tons per capita)': {'realistic_min': 0.1, 'realistic_max': 20},
            'Population, total': {'realistic_min': 1e6, 'realistic_max': 1e9}
        }

        for indicator, ranges in key_indicators.items():
            prophet_col = f"{indicator}_Prophet"
            if prophet_col in forecasts.columns:
                if forecasts[prophet_col].notna().any():
                    min_val = forecasts[prophet_col][forecasts[prophet_col].notna()].min()
                    max_val = forecasts[prophet_col][forecasts[prophet_col].notna()].max()
                    print(f"{prophet_col}: range {min_val} to {max_val}")

                    # Flag potentially problematic values
                    if min_val < ranges['realistic_min'] or max_val > ranges['realistic_max']:
                        print(f"WARNING: Values for {prophet_col} might be outside realistic range!")
                        validation_passed = False
                else:
                    print(f"WARNING: {prophet_col} exists but contains only null values")
                    validation_passed = False
            else:
                print(f"WARNING: Required indicator {prophet_col} not found")
                validation_passed = False

        # Return validation result
        if validation_passed:
            print("Validation PASSED: Forecast data appears to be suitable for analysis")
        else:
            print("Validation WARNING: Forecast data may have issues. Results should be carefully reviewed.")

        return validation_passed

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

        # Validate forecast data
        self.validate_forecast_data(forecasts, country)

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
            # Always prioritize Prophet forecasts when available
            prophet_column = f"{base_indicator}_Prophet"
            if prophet_column in future_data.columns and future_data[prophet_column].notna().any():
                print(f"    Using Prophet forecast column '{prophet_column}' for {base_indicator}")
                return prophet_column

            # Only fall back to other columns if Prophet isn't available
            variants = [
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
                [(1 + 0.002) ** (y - base_year) for y in years]
            )

            # Initialize results dictionary
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
        Calculate economic and environmental benefits of efficiency scenarios
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
                'current': self.energy_cost_per_mwh,
                'projected': [self.energy_cost_per_mwh * (1 + self.energy_price_increase) ** (y - min(years)) for y in years]
            }

            # Environmental damage cost (USD per ton of CO2)
            env_damage_cost = {
                'current': self.carbon_damage_cost,
                'projected': [self.carbon_damage_cost * (1 + self.carbon_damage_increase) ** (y - min(years)) for y in years]
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
                job_creation = np.array([data['implementation_cost'][i] * 0.00005 for i in range(len(years))])
                health_benefits = env_damage_avoided * 0.3  # 30% of environmental benefits are health-related

                # Additional program benefits
                energy_security_benefit = energy_savings_twh * 10  # $10/MWh security premium

                # Adjust implementation costs for economies of scale
                adjusted_implementation_cost = data['implementation_cost'] * self.implementation_cost_factor

                # Net economic benefit
                net_benefit = (energy_cost_savings +
                               env_damage_avoided +
                               energy_security_benefit -
                               adjusted_implementation_cost)

                # Payback period calculation
                cumulative_cost = np.cumsum(adjusted_implementation_cost)
                cumulative_benefit = np.cumsum(energy_cost_savings + env_damage_avoided + energy_security_benefit)

                payback_period = None
                for i in range(len(years)):
                    if cumulative_benefit[i] >= cumulative_cost[i]:
                        if i == 0:
                            # For first year, estimate partial year payback
                            payback_period = cumulative_cost[i] / cumulative_benefit[i]
                        else:
                            # Interpolate between years
                            fraction = (cumulative_cost[i-1] - cumulative_benefit[i-1]) / (
                                    cumulative_benefit[i] - cumulative_benefit[i-1] - (
                                    cumulative_cost[i] - cumulative_cost[i-1]))
                            payback_period = (i - 1) + fraction
                        break

                # ROI calculation (Return on Investment)
                total_investment = np.sum(adjusted_implementation_cost)
                total_benefit = np.sum(energy_cost_savings + env_damage_avoided + energy_security_benefit)

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
                print(f"    Original implementation cost: ${np.sum(data['implementation_cost']) / 1e9:.2f} billion")
                print(f"    Adjusted implementation cost: ${np.sum(adjusted_implementation_cost) / 1e9:.2f} billion")
                print(f"    Total CO2 reduction: {np.sum(total_co2_reduction) / 1e6:.2f} million tons")
                print(f"    ROI: {roi * 100:.1f}%")
                print(
                    f"    Payback period: {payback_period:.1f} years" if payback_period else "    Payback period: Beyond timeframe")

                # Store results
                benefits['economic'][scenario] = {
                    'energy_savings_twh': energy_savings_twh,
                    'energy_cost_savings': energy_cost_savings,
                    'implementation_cost': adjusted_implementation_cost,  # Using adjusted cost
                    'original_implementation_cost': data['implementation_cost'],  # Keep original for reference
                    'energy_security_benefit': energy_security_benefit,
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
        Calculate potential carbon credits from efficiency improvements
        """
        print(f"  Calculating potential carbon credits for {country}")

        carbon_credits = {}
        years = results['years']

        # Determine if we're analyzing Thailand
        is_thailand = country.lower() == 'thailand'

        if is_thailand:
            print("  Using carbon market data for calculations")
            # Create sector-specific carbon credit projections
            sectors = {
                'agriculture': {
                    'share': 0.20,  # 20% from agricultural sustainability projects
                    'price': self.carbon_market.prices['agriculture'],
                    'growth_rate': self.carbon_market.annual_price_growth
                },
                'solar': {
                    'share': 0.35,  # 35% from solar PV
                    'price': self.carbon_market.prices['solar'],
                    'growth_rate': self.carbon_market.annual_price_growth * 1.1
                },
                'industrial': {
                    'share': 0.25,  # 25% from industrial EE
                    'price': self.carbon_market.prices['industrial'],
                    'growth_rate': self.carbon_market.annual_price_growth
                },
                'building': {
                    'share': 0.20,  # 20% from LED streetlighting and building EE
                    'price': self.carbon_market.prices['building'],
                    'growth_rate': self.carbon_market.annual_price_growth
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

            print(f"  Carbon price trajectory: ${carbon_prices[0]:.2f} (2024) to ${carbon_prices[-1]:.2f} (2035)")
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

        # Apply crediting factor - not all emissions reductions generate carbon credits
        crediting_factor = self.crediting_factor  # From program configuration

        # Calculate credits for each scenario
        for scenario, data in results['scenarios'].items():
            co2_reduction = results['bau']['co2'] - data['co2']
            total_co2_reduction = co2_reduction * results['bau']['population']

            # Apply crediting factor
            creditable_co2 = total_co2_reduction * crediting_factor

            # Calculate carbon credit value
            carbon_credit_value = np.array([creditable_co2[i] * carbon_prices[i]
                                            for i in range(len(years))])

            # Calculate management fees
            management_fee = carbon_credit_value * self.carbon_market.management_fee_percent
            net_credit_value = carbon_credit_value - management_fee

            # Calculate cumulative credit generation
            cumulative_credits = np.cumsum(creditable_co2)
            cumulative_value = np.cumsum(carbon_credit_value)

            # Calculate aggregation potential (combining small projects into larger portfolios)
            aggregation_premium = self.carbon_market.aggregation_premium
            aggregation_potential = carbon_credit_value * (1 + aggregation_premium)

            # Calculate sector breakdown for Thailand
            sector_breakdown = None
            if is_thailand:
                sector_breakdown = []
                for i in range(len(years)):
                    year_breakdown = {}
                    for sector, info in sectors.items():
                        # Calculate credits by sector
                        sector_credits = creditable_co2[i] * info['share']
                        sector_value = sector_credits * sector_prices[i][sector]
                        year_breakdown[sector] = {
                            'credits': sector_credits,
                            'value': sector_value,
                            'price': sector_prices[i][sector]
                        }
                    sector_breakdown.append(year_breakdown)

            # Calculate registration and verification costs
            registration_cost = carbon_credit_value * self.carbon_market.registration_cost_percent
            verification_cost = carbon_credit_value * self.carbon_market.verification_cost_percent
            total_cost = registration_cost + verification_cost

            carbon_credits[scenario] = {
                'annual_credits': creditable_co2,  # Using creditable CO2 instead of total
                'total_co2_reduction': total_co2_reduction,  # Keep total for reference
                'annual_value': carbon_credit_value,
                'management_fee': management_fee,
                'net_value': net_credit_value,
                'cumulative_credits': cumulative_credits,
                'cumulative_value': cumulative_value,
                'aggregation_potential': aggregation_potential,
                'carbon_prices': carbon_prices,
                'crediting_factor': crediting_factor,
                'registration_cost': registration_cost,
                'verification_cost': verification_cost,
                'total_cost': total_cost
            }

            # Add sector-specific information
            if is_thailand:
                carbon_credits[scenario].update({
                    'sector_breakdown': sector_breakdown
                })

        return carbon_credits

    def _calculate_loan_repayment(self, country, results, benefits, carbon_credits):
        """
        Calculate loan repayment schedule for financing program

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

        Returns:
        --------
        dict
            Loan repayment analysis
        """
        if country.lower() != 'thailand':
            return None

        print(f"  Calculating loan repayment for {country}")

        # Program loan amount
        loan_amount = self.loan_amount  # From program configuration

        years = results['years']
        repayment_analysis = {}

        # Loan terms
        annual_interest_rate = self.loan_interest_rate
        grace_period_years = self.loan_grace_period

        # Revenue allocation to loan repayment
        energy_savings_allocation = self.energy_savings_allocation
        carbon_allocation = self.carbon_revenue_allocation

        # Calculate for each scenario
        for scenario in benefits['economic']:
            # Get energy cost savings and carbon revenue
            energy_savings = benefits['economic'][scenario]['energy_cost_savings']
            carbon_revenue = carbon_credits[scenario]['net_value']  # After management fee

            # Calculate payment streams
            energy_payment = energy_savings * energy_savings_allocation
            carbon_payment = carbon_revenue * carbon_allocation

            # Calculate total annual payment
            total_payment = energy_payment + carbon_payment

            # Calculate remaining balance with interest
            remaining_balance = np.zeros(len(years))

            # Apply grace period (interest accrues but no principal payments)
            remaining_balance[0] = loan_amount

            for i in range(1, min(grace_period_years, len(years))):
                interest = remaining_balance[i-1] * annual_interest_rate
                remaining_balance[i] = remaining_balance[i-1] + interest

            # Calculate repayment after grace period
            for i in range(grace_period_years, len(years)):
                if i >= len(years):
                    break

                interest = remaining_balance[i-1] * annual_interest_rate
                payment = total_payment[i]
                remaining_balance[i] = remaining_balance[i-1] + interest - payment
                if remaining_balance[i] < 0:
                    remaining_balance[i] = 0

            # Find payback period
            payback_period = None
            payback_year = None

            for i, year in enumerate(years):
                if i >= grace_period_years and remaining_balance[i] <= 0:
                    if i > grace_period_years and remaining_balance[i-1] > 0:
                        # Interpolate for more precise estimate
                        prev_balance = remaining_balance[i-1]
                        # Calculate balance reduction in this period
                        interest = prev_balance * annual_interest_rate
                        balance_reduction = prev_balance + interest - total_payment[i]
                        if balance_reduction < 0:
                            # Calculate what fraction of the year is needed
                            fraction = prev_balance / (total_payment[i] - interest)
                            payback_period = i - grace_period_years + fraction
                            payback_year = years[i-1] + fraction
                        else:
                            payback_period = i - grace_period_years
                            payback_year = years[i]
                    else:
                        payback_period = i - grace_period_years
                        payback_year = years[i]
                    break

            # If no payback within the period, estimate remaining years
            if payback_period is None and remaining_balance[-1] > 0:
                # Estimate years beyond our timeframe
                annual_payment_avg = np.mean(total_payment[-3:])  # Average of last 3 years
                remaining_with_interest = remaining_balance[-1] * (1 + annual_interest_rate)
                if annual_payment_avg > remaining_balance[-1] * annual_interest_rate:
                    years_beyond = remaining_balance[-1] / (annual_payment_avg - (remaining_balance[-1] * annual_interest_rate))
                    payback_period = len(years) - grace_period_years + years_beyond
                    payback_year = years[-1] + years_beyond
                else:
                    payback_period = None
                    payback_year = None

            # Store results
            repayment_analysis[scenario] = {
                'years': years,
                'energy_payment': energy_payment,
                'carbon_payment': carbon_payment,
                'total_payment': total_payment,
                'remaining_balance': remaining_balance,
                'payback_period': payback_period,
                'payback_year': payback_year,
                'loan_amount': loan_amount,
                'interest_rate': annual_interest_rate,
                'grace_period': grace_period_years,
                'allocation': {
                    'energy_savings': energy_savings_allocation,
                    'carbon_credits': carbon_allocation
                }
            }

            # Print summary
            if payback_period is not None:
                energy_total = np.sum(energy_payment) / 1e6
                carbon_total = np.sum(carbon_payment) / 1e6
                total = energy_total + carbon_total
                energy_pct = (energy_total / total) * 100 if total > 0 else 0
                carbon_pct = (carbon_total / total) * 100 if total > 0 else 0

                print(f"    {scenario.capitalize()} scenario:")
                print(f"    Payback period: {payback_period:.2f} years (by {payback_year:.2f})")
                print(f"    Energy payments: ${energy_total:.2f} million ({energy_pct:.1f}%)")
                print(f"    Carbon payments: ${carbon_total:.2f} million ({carbon_pct:.1f}%)")
            else:
                print(f"    {scenario.capitalize()} scenario does not reach payback within timeframe")

        return repayment_analysis

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

        # 1. Energy Demand Scenarios
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

        # 2. CO2 Emissions Scenarios
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
            title += " (Including Carbon Revenue)"

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
            title += " (Projected)"

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
        Create Thailand-specific visualizations for carbon market analysis

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

        print(f"  Creating additional market visualizations")

        output_dir = os.path.join('output', country, 'efficiency_analysis', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        years = results['years']

        # 1. Sector Value Comparison
        plt.figure(figsize=(12, 8))

        # Use medium scenario as an example
        scenario = 'medium'
        if scenario in carbon_credits and 'sector_breakdown' in carbon_credits[scenario]:
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

            plt.title(f"Carbon Credit Value by Sector ({scenario.capitalize()} Scenario)", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Carbon Credit Value (Million USD)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sector_value.png'), dpi=300)
            plt.close()

        # 2. Price Trajectory
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

            plt.title(f"Carbon Price Projections by Sector", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Carbon Price (USD/tonne CO2)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'price_projections.png'), dpi=300)
            plt.close()

        # 3. Cost comparison - costs vs revenue
        plt.figure(figsize=(12, 6))

        if scenario in carbon_credits:
            registration_costs = carbon_credits[scenario]['registration_cost'] / 1e6
            verification_costs = carbon_credits[scenario]['verification_cost'] / 1e6
            management_fees = carbon_credits[scenario]['management_fee'] / 1e6
            credit_revenue = carbon_credits[scenario]['annual_value'] / 1e6
            net_revenue = carbon_credits[scenario]['net_value'] / 1e6

            # Create stacked bar for costs and a line for revenue
            plt.bar(years, registration_costs, label="Registration Costs", color='#CD5C5C', alpha=0.7)
            plt.bar(years, verification_costs, bottom=registration_costs, label="Verification Costs", color='#F08080',
                    alpha=0.7)
            plt.bar(years, management_fees, bottom=registration_costs+verification_costs,
                    label="Management Fees", color='#FA8072', alpha=0.7)
            plt.plot(years, credit_revenue, 'k-', linewidth=2, label="Gross Revenue")
            plt.plot(years, net_revenue, 'g--', linewidth=2, label="Net Revenue")

            plt.title(f"Carbon Credit Costs vs Revenue ({scenario.capitalize()} Scenario)", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Million USD", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'costs_vs_revenue.png'), dpi=300)
            plt.close()

        # 4. Loan Repayment Visualization
        plt.figure(figsize=(12, 6))

        # Assuming loan repayment data is passed to this function
        # For now, let's just create a placeholder plot to be updated later

        plt.title("Loan Repayment Projections", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Million USD", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loan_repayment.png'), dpi=300)
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
                    "The carbon market offers excellent opportunities for carbon credit generation:\n\n")
                f.write("- **Current Carbon Prices by Sector:**\n")
                f.write(f"  - Agriculture/Forestry: ${self.carbon_market.prices['agriculture']} per ton CO2\n")
                f.write(f"  - Solar Energy: ${self.carbon_market.prices['solar']} per ton CO2\n")
                f.write(f"  - Industrial: ${self.carbon_market.prices['industrial']} per ton CO2\n")
                f.write(f"  - Building: ${self.carbon_market.prices['building']} per ton CO2\n")
                f.write(f"  - Market Average: ${self.carbon_market.prices['average']} per ton CO2\n\n")
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

    def _generate_program_report(self, country, forecasts, results, benefits, carbon_credits, loan_repayment):
        """
        Generate a comprehensive report for the financing program

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
        loan_repayment : dict
            Loan repayment analysis
        """
        print(f"  Generating program report for {country}")

        output_dir = os.path.join('output', country, 'efficiency_analysis')
        report_file = os.path.join(output_dir, 'financing_program_report.md')

        with open(report_file, 'w') as f:
            f.write(f"# Low-Carbon City Financing Program - {country}\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n")

            f.write("## Executive Summary\n\n")

            # Use medium scenario as default for summary
            scenario = 'medium'

            if scenario in carbon_credits and scenario in benefits['economic']:
                roi = benefits['economic'][scenario]['roi'] * 100
                payback = loan_repayment[scenario]['payback_period'] if loan_repayment and scenario in loan_repayment else None

                total_co2 = np.sum(carbon_credits[scenario]['annual_credits']) / 1e6
                carbon_value = np.sum(carbon_credits[scenario]['annual_value']) / 1e6
                management_fee = carbon_value * self.carbon_market.management_fee_percent  # 10% management fee
                net_carbon_value = carbon_value - management_fee

                energy_savings = np.sum(benefits['economic'][scenario]['energy_savings_twh'])
                energy_savings_value = np.sum(benefits['economic'][scenario]['energy_cost_savings']) / 1e9

                f.write(f"The proposed low-carbon city financing program for {country} demonstrates significant potential for both emissions reductions and economic benefits:\n\n")

                f.write(f"- **Program Loan Amount:** ${self.loan_amount/1e6:.1f} million\n")
                f.write(f"- **Total CO₂ Reduction:** {total_co2:.2f} million tons\n")
                f.write(f"- **Carbon Credit Value:** ${carbon_value:.2f} million\n")
                f.write(f"- **Management Fee:** ${management_fee:.2f} million\n")
                f.write(f"- **Net Carbon Revenue:** ${net_carbon_value:.2f} million\n")
                f.write(f"- **Energy Savings:** {energy_savings:.2f} TWh (${energy_savings_value:.2f} billion)\n")
                f.write(f"- **Return on Investment:** {roi:.1f}%\n")

                if payback:
                    f.write(f"- **Loan Repayment Period:** {payback:.1f} years\n\n")
                else:
                    f.write(f"- **Loan Repayment:** Beyond program timeframe\n\n")

            # Project Structure & Components
            f.write("## Program Structure & Components\n\n")

            f.write("### Financing Modality\n\n")
            f.write(f"- ${self.loan_amount/1e6:.1f} million loan for implementation of energy efficiency and renewable energy projects\n")
            f.write("- Focus on large-scale public infrastructure and community-level initiatives\n")
            f.write("- Disbursement linked to verified emissions reductions and project milestones\n\n")

            f.write("### Implementation Structure\n\n")
            f.write("- **Public Sector Organizations:** Implementing large-scale infrastructure projects\n")
            f.write("- **Local Financial Institutions:** Financing community-level projects\n")
            f.write("- **Coordinating Entity:** Aggregating and monetizing carbon credits\n")
            f.write("- **Program Management:** Oversight, MRV, and compliance\n\n")

            # Carbon Market Integration
            f.write("## Carbon Market Integration\n\n")

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

            # Loan Repayment Analysis
            if loan_repayment:
                f.write("## Loan Repayment Analysis\n\n")

                f.write("The program integrates carbon credit revenues with energy cost savings to accelerate loan repayment.\n\n")

                for scenario_name, repayment in loan_repayment.items():
                    f.write(f"### {scenario_name.capitalize()} Scenario\n\n")

                    if repayment['payback_period'] is not None:
                        f.write(f"- **Loan Amount:** ${repayment['loan_amount']/1e6:.1f} million\n")
                        f.write(f"- **Interest Rate:** {repayment['interest_rate']*100:.1f}%\n")
                        f.write(f"- **Grace Period:** {repayment['grace_period']} years\n")
                        f.write(f"- **Payback Period:** {repayment['payback_period']:.2f} years\n")
                        f.write(f"- **Payback Completion:** Year {repayment['payback_year']:.1f}\n")

                        total_energy = np.sum(repayment['energy_payment']) / 1e6
                        total_carbon = np.sum(repayment['carbon_payment']) / 1e6
                        total_payment = total_energy + total_carbon

                        f.write(f"- **Total Energy Savings Payment:** ${total_energy:.2f} million ({(total_energy / total_payment * 100):.1f}%)\n")
                        f.write(f"- **Total Carbon Credit Payment:** ${total_carbon:.2f} million ({(total_carbon / total_payment * 100):.1f}%)\n")
                        f.write(f"- **Total Repayment Amount:** ${total_payment:.2f} million\n\n")
                    else:
                        f.write("- No complete payback within the analysis timeframe.\n\n")

                    f.write("#### Annual Repayment Schedule\n\n")
                    f.write("| Year | Energy Payment ($M) | Carbon Payment ($M) | Total Payment ($M) | Remaining Balance ($M) |\n")
                    f.write("|------|---------------------|---------------------|--------------------|-----------------------|\n")

                    for i, year in enumerate(repayment['years']):
                        energy_payment = repayment['energy_payment'][i] / 1e6
                        carbon_payment = repayment['carbon_payment'][i] / 1e6
                        total_payment = repayment['total_payment'][i] / 1e6
                        remaining = repayment['remaining_balance'][i] / 1e6

                        f.write(f"| {year} | ${energy_payment:.2f} | ${carbon_payment:.2f} | ${total_payment:.2f} | ${max(0, remaining):.2f} |\n")

                    f.write("\n")

            # NDC Contribution
            f.write("## Climate Policy Contribution\n\n")

            f.write("The program will significantly contribute to climate mitigation targets and policy objectives:\n\n")

            # Calculate contribution based on 2030 values
            if 'high' in carbon_credits and 'bau' in results and 2030 in results['years']:
                idx_2030 = list(results['years']).index(2030)
                bau_emissions = results['bau']['co2'][idx_2030] * results['bau']['population'][idx_2030]

                for scenario, data in results['scenarios'].items():
                    scenario_emissions = data['co2'][idx_2030] * results['bau']['population'][idx_2030]
                    reduction_percentage = (bau_emissions - scenario_emissions) / bau_emissions * 100
                    f.write(f"- **{scenario.capitalize()} Scenario:** {reduction_percentage:.1f}% reduction from BAU by 2030\n")

                f.write("\nThese reductions can be used for:\n")
                f.write("- Fulfilling national climate commitments\n")
                f.write("- Supporting carbon neutrality goals\n")
                f.write("- Developing domestic carbon market infrastructure\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            # Find best scenario based on ROI or payback
            best_scenario = None
            if loan_repayment:
                payback_periods = {s: loan_repayment[s]['payback_period'] for s in loan_repayment if loan_repayment[s]['payback_period']}
                if payback_periods:
                    best_scenario = min(payback_periods, key=payback_periods.get)

            if best_scenario:
                f.write(f"1. **Implementation Strategy:** The {best_scenario.capitalize()} efficiency scenario offers the best balance of climate impact and financial sustainability.\n")
            else:
                # Fall back to ROI comparison
                rois = {s: benefits['economic'][s]['roi'] for s in benefits['economic']}
                best_scenario = max(rois, key=rois.get)
                f.write(f"1. **Implementation Strategy:** The {best_scenario.capitalize()} efficiency scenario offers the best return on investment.\n")

            f.write("2. **Credit Aggregation:** Aggregating carbon credits from multiple projects increases marketability and value.\n")
            f.write("3. **Sector Prioritization:** Focus on sectors with highest carbon credit value and implementation feasibility.\n")
            f.write("4. **MRV Systems:** Implement robust monitoring and verification to maximize credit issuance.\n")
            f.write("5. **Market Engagement:** Actively engage with international carbon credit buyers to secure premium prices.\n\n")

            # Conclusion
            f.write("## Conclusion\n\n")
            f.write(f"The low-carbon city financing program presents a strategic approach to accelerating climate action in {country}. By monetizing carbon credits and capturing energy cost savings, the program creates a sustainable financing mechanism that can transform urban infrastructure while meeting climate goals.\n\n")
            f.write("The program's integrated approach leverages international climate finance while building domestic capacity for low-carbon development. With proper implementation and carbon market integration, the initiative can deliver substantial environmental benefits while maintaining financial sustainability.\n\n")
            f.write("*Note: This analysis is based on forecasted data and current market conditions. Results will vary based on implementation effectiveness and market developments.*\n")

        print(f"  Report saved to {report_file}")

    def analyze_country(self, country):
        """
        Perform energy efficiency analysis for a specific country with financing program integration

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

            # Validate forecast data
            validation_passed = self.validate_forecast_data(forecasts, country)
            if not validation_passed:
                print("WARNING: Proceeding with analysis despite validation warnings. Results may be unreliable.")

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

            # Calculate loan repayment for financing program
            loan_repayment = None
            if is_thailand:
                loan_repayment = self._calculate_loan_repayment(country, results, benefits, carbon_credits)

            # Generate standard visualizations
            self._create_visualizations(country, forecasts, results, benefits, carbon_credits)

            # For Thailand, generate additional visualizations
            if is_thailand:
                self._create_thailand_specific_visualizations(country, results, benefits, carbon_credits)

            # Generate standard report
            self._generate_report(country, forecasts, results, benefits, carbon_credits)

            # Generate financing program report for Thailand
            if is_thailand:
                self._generate_program_report(country, forecasts, results, benefits, carbon_credits, loan_repayment)

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

    print("Energy Efficiency Analyzer")
    print("--------------------------------")

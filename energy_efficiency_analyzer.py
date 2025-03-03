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

        # Appliance lifetime and turnover rates
        self.appliance_turnover = {
            'residential_ac': {
                'lifetime_years': 10,
                'replacement_rate': 0.09,  # 9% per year
                'efficiency_gain_per_replacement': 0.15  # 15% improvement with new unit
            },
            'commercial_ac': {
                'lifetime_years': 15,
                'replacement_rate': 0.06,  # 6% per year
                'efficiency_gain_per_replacement': 0.12  # 12% improvement with new unit
            },
            'refrigerators': {
                'lifetime_years': 12,
                'replacement_rate': 0.08,  # 8% per year
                'efficiency_gain_per_replacement': 0.10  # 10% improvement with new unit
            },
            'lighting': {
                'lifetime_years': 5,
                'replacement_rate': 0.18,  # 18% per year
                'efficiency_gain_per_replacement': 0.25  # 25% improvement with new unit (LED adoption)
            }
        }

        # Climate change impact factors
        self.climate_factors = {
            'cooling_demand_increase': 0.006,  # 0.6% increase per year
            'temperature_increase': 0.003,     # 0.3% increase per year
            'extreme_weather_cost': 0.003,     # 0.3% of GDP impact
            'ac_efficiency_improvement': 0.015, # 1.5% natural efficiency improvement per year
            'technology_learning_rate': 0.12    # 12% cost reduction per doubling of capacity
        }

        # Adaptive scenarios with S-curve adoption for Thailand
        self.adaptive_scenarios = {
            'low': {
                'initial_adoption_rate': 0.02,    # 2% initial adoption rate
                'max_adoption_rate': 0.05,        # 5% max annual adoption rate
                'saturation_level': 0.65,         # 65% market saturation
                'appliance_turnover_factor': 1.0,  # Standard appliance turnover
                'demand_reduction_max': 0.18,     # 18% max demand reduction
                'co2_reduction_max': 0.25,        # 25% max CO2 reduction
                'implementation_cost': 0.008      # 0.8% of GDP
            },
            'medium': {
                'initial_adoption_rate': 0.03,    # 3% initial adoption rate
                'max_adoption_rate': 0.08,        # 8% max annual adoption rate
                'saturation_level': 0.80,         # 80% market saturation
                'appliance_turnover_factor': 1.2,  # 20% accelerated turnover
                'demand_reduction_max': 0.32,     # 32% max demand reduction
                'co2_reduction_max': 0.45,        # 45% max CO2 reduction
                'implementation_cost': 0.015      # 1.5% of GDP
            },
            'high': {
                'initial_adoption_rate': 0.05,    # 5% initial adoption rate
                'max_adoption_rate': 0.12,        # 12% max annual adoption rate
                'saturation_level': 0.90,         # 90% market saturation
                'appliance_turnover_factor': 1.5,  # 50% accelerated turnover
                'demand_reduction_max': 0.50,     # 50% max demand reduction
                'co2_reduction_max': 0.65,        # 65% max CO2 reduction
                'implementation_cost': 0.025      # 2.5% of GDP
            }
        }

        # Traditional efficiency scenarios (for countries other than Thailand)
        self.efficiency_scenarios = efficiency_scenarios or {
            'low': {
                'demand_reduction': 0.12,  # 12% reduction (beyond natural efficiency improvements)
                'co2_reduction': 0.18,     # 18% CO2 reduction
                'implementation_cost': 0.008  # 0.8% of GDP
            },
            'medium': {
                'demand_reduction': 0.25,  # 25% reduction (beyond natural efficiency improvements)
                'co2_reduction': 0.35,     # 35% CO2 reduction
                'implementation_cost': 0.015  # 1.5% of GDP
            },
            'high': {
                'demand_reduction': 0.40,  # 40% reduction (beyond natural efficiency improvements)
                'co2_reduction': 0.55,     # 55% CO2 reduction
                'implementation_cost': 0.025  # 2.5% of GDP
            }
        }

        # Add carbon market information
        self.carbon_market = CarbonMarketInfo()

        # Generic carbon credit pricing (used for non-Thailand countries)
        self.carbon_credit_pricing = {
            'current': 8,      # Current price
            'projected_2030': 15,  # 2030 price
            'projected_2035': 25   # 2035 price
        }

        # Thailand's historical energy efficiency investments & programs
        self.thailand_ee_investments = {
            # Historical investments in million USD
            'historical': {
                '2015-2020': 450,  # Energy Efficiency Development Plan investments
                '2020-2023': 680,  # Recent EE investments
            },
            # Types of energy efficiency programs
            'program_types': {
                'building_codes': 0.25,       # 25% of investment
                'appliance_standards': 0.30,  # 30% of investment
                'industrial_programs': 0.20,  # 20% of investment
                'awareness_campaigns': 0.10,  # 10% of investment
                'utility_programs': 0.15      # 15% of investment
            }
        }

        # Sample data for Thailand's historical EE programs
        self.thailand_ee_data = {
            'historical_investments': pd.DataFrame({
                'Year': range(2010, 2024),
                'Investment_Million_USD': [120, 150, 180, 200, 220, 250, 280, 300, 320, 350, 380, 420, 450, 480]
            }),
            'energy_savings': pd.DataFrame({
                'Year': range(2010, 2024),
                'Savings_GWh': [200, 350, 500, 750, 1000, 1250, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
            }),
            'program_breakdown': {
                'Building_Codes': 0.26,
                'Appliance_Standards': 0.31,
                'Industrial_Programs': 0.19,
                'Awareness_Campaigns': 0.09,
                'Utility_Programs': 0.15
            }
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

    def calculate_s_curve_adoption(self, years, scenario_params):
        """
        Calculate S-curve adoption rate over time

        Parameters:
        -----------
        years : array-like
            Array of years to calculate adoption for
        scenario_params : dict
            Parameters for the scenario (initial_adoption_rate, max_adoption_rate, saturation_level)

        Returns:
        --------
        array
            Adoption rates for each year
        """
        # Parameters for the logistic S-curve
        initial_rate = scenario_params['initial_adoption_rate']
        max_rate = scenario_params['max_adoption_rate']
        saturation = scenario_params['saturation_level']

        # Midpoint of the years array (for S-curve centering)
        mid_year = (years[-1] + years[0]) / 2

        # Steepness of the curve
        k = 0.5

        # Calculate S-curve for adoption rates
        adoption_rates = []
        cumulative_adoption = 0

        for year in years:
            # Logistic S-curve formula
            s_curve_factor = 1 / (1 + np.exp(-k * (year - mid_year)))

            # Calculate adoption rate for this year (varies between initial and max)
            adoption_rate = initial_rate + (max_rate - initial_rate) * s_curve_factor

            # Adjust for approaching saturation
            if cumulative_adoption > 0:
                # Reduce adoption as we approach saturation
                saturation_factor = max(0, 1 - (cumulative_adoption / saturation))
                adoption_rate *= saturation_factor

            # Add to cumulative adoption (but don't exceed saturation)
            cumulative_adoption = min(cumulative_adoption + adoption_rate, saturation)
            adoption_rates.append(adoption_rate)

        return np.array(adoption_rates)

    def calculate_cumulative_adoption(self, adoption_rates):
        """Calculate cumulative adoption from annual adoption rates"""
        return np.cumsum(adoption_rates)

    def calculate_natural_efficiency_improvement(self, years, base_year):
        """
        Calculate natural efficiency improvement from appliance turnover

        Parameters:
        -----------
        years : array-like
            Array of years to calculate for
        base_year : int
            Base year for calculations

        Returns:
        --------
        array
            Efficiency improvement factors for each year
        """
        efficiency_improvements = []

        for year in years:
            years_passed = year - base_year

            # Calculate weighted average efficiency improvement from appliance turnover
            weighted_improvement = 0
            total_weight = 0

            for appliance, data in self.appliance_turnover.items():
                # Calculate how many replacement cycles have occurred
                replacement_cycles = years_passed * data['replacement_rate']
                # Efficiency gain from replacements
                efficiency_gain = 1 - (1 - data['efficiency_gain_per_replacement']) ** replacement_cycles

                # Weight by replacement rate (proxy for importance in energy consumption)
                weighted_improvement += efficiency_gain * data['replacement_rate']
                total_weight += data['replacement_rate']

            # Normalize by total weight
            if total_weight > 0:
                avg_efficiency_improvement = weighted_improvement / total_weight
            else:
                avg_efficiency_improvement = 0

            efficiency_improvements.append(avg_efficiency_improvement)

        return np.array(efficiency_improvements)
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

        # Check if this is Thailand (use enhanced model)
        is_thailand = (country.lower() == 'thailand')

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

            # Fill any null values
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
                'demand': demand_data.values,
                'co2': co2_data.values,
                'gdp': gdp_data.values,
                'population': pop_data.values
            }

            # Print data preview
            print("\n  Data preview:")
            for key, data in baseline_values.items():
                print(f"    {key.capitalize()}: min={np.min(data)}, max={np.max(data)}, mean={np.mean(data)}")

            # Get years and base year
            years = future_data['Year'].values
            base_year = min(years)

            # Calculate natural efficiency improvement from appliance turnover
            natural_efficiency = self.calculate_natural_efficiency_improvement(years, base_year)

            # Calculate BAU with climate effects and natural efficiency improvement
            bau_demand = np.array(baseline_values['demand'].copy())

            # Apply climate factors and natural efficiency improvement
            for i, year in enumerate(years):
                years_passed = year - base_year

                # Climate change increases cooling demand
                climate_multiplier = (1 + self.climate_factors['cooling_demand_increase']) ** years_passed

                # Natural efficiency improvements offset some of this
                if is_thailand:
                    # Use detailed natural efficiency model for Thailand
                    efficiency_multiplier = 1 - natural_efficiency[i]
                else:
                    # Simpler model for other countries
                    natural_efficiency_improvement = self.climate_factors['ac_efficiency_improvement'] * years_passed
                    efficiency_multiplier = max(0.7, 1 - natural_efficiency_improvement)  # Cap at 30% improvement

                # Apply both factors to BAU demand
                bau_demand[i] *= climate_multiplier * efficiency_multiplier

            # Apply modest CO2 increase due to climate change (less efficiency at higher temps)
            bau_co2 = baseline_values['co2'] * np.array(
                [(1 + 0.002) ** (y - base_year) for y in years]
            )

            # Initialize results dictionary
            results = {
                'years': years,
                'bau': {  # Business as usual with climate impacts
                    'demand': bau_demand,
                    'co2': bau_co2,
                    'gdp': baseline_values['gdp'],
                    'population': baseline_values['population']
                },
                'natural_efficiency': natural_efficiency,
                'scenarios': {}
            }

            # Use Thailand-specific adaptive scenarios or regular scenarios
            if is_thailand:
                # Generate Thailand scenarios with S-curve adoption
                scenarios = self.adaptive_scenarios

                for scenario_name, params in scenarios.items():
                    # Calculate S-curve adoption rates
                    adoption_rates = self.calculate_s_curve_adoption(years, params)
                    cumulative_adoption = self.calculate_cumulative_adoption(adoption_rates)

                    # Calculate effect of accelerated appliance turnover
                    turnover_factor = params['appliance_turnover_factor']
                    if turnover_factor > 1.0:
                        # Calculate improved efficiency from accelerated turnover
                        accelerated_efficiency = 1 - (1 - natural_efficiency) * turnover_factor
                    else:
                        accelerated_efficiency = natural_efficiency

                    # Calculate demand reduction (increases over time with S-curve)
                    max_demand_reduction = params['demand_reduction_max']
                    demand_reduction = max_demand_reduction * cumulative_adoption

                    # Apply demand reduction to BAU
                    scenario_demand = bau_demand * (1 - demand_reduction)

                    # Calculate CO2 reduction (increases over time with S-curve)
                    max_co2_reduction = params['co2_reduction_max']
                    co2_reduction = max_co2_reduction * cumulative_adoption

                    # Apply CO2 reduction to BAU
                    scenario_co2 = bau_co2 * (1 - co2_reduction)

                    # Calculate implementation costs (front-loaded for policy implementation)
                    base_implementation_cost = params['implementation_cost']
                    # Cost curve is higher in early years then declines with learning
                    implementation_cost_curve = base_implementation_cost * (
                            1 - np.array([min(0.5, self.climate_factors['technology_learning_rate'] *
                                              np.log2(1 + i)) for i in range(len(years))])
                    )
                    implementation_costs = results['bau']['gdp'] * implementation_cost_curve

                    # Store scenario results
                    results['scenarios'][scenario_name] = {
                        'demand': scenario_demand,
                        'co2': scenario_co2,
                        'implementation_cost': implementation_costs,
                        'adoption_rates': adoption_rates,
                        'cumulative_adoption': cumulative_adoption,
                        'demand_reduction': demand_reduction,
                        'co2_reduction': co2_reduction,
                        'params': params
                    }

                    # Print scenario summary
                    print(f"\n  {scenario_name.capitalize()} scenario:")
                    print(f"    Max demand reduction: {max_demand_reduction * 100:.1f}%")
                    print(f"    Max CO2 reduction: {max_co2_reduction * 100:.1f}%")
                    print(f"    Final adoption level: {cumulative_adoption[-1] * 100:.1f}%")
            else:
                # Generate standard scenarios for other countries
                for scenario, params in self.efficiency_scenarios.items():
                    demand_reduction = params['demand_reduction']
                    co2_reduction = params['co2_reduction']
                    implementation_cost = params['implementation_cost']

                    # Apply reductions to BAU (calculate departures from BAU)
                    improved_demand = bau_demand * (1 - demand_reduction)
                    improved_co2 = bau_co2 * (1 - co2_reduction)

                    # Calculate implementation costs
                    implementation_costs = results['bau']['gdp'] * implementation_cost

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
            is_thailand = (country.lower() == 'thailand')

            # Energy cost assumptions (USD per MWh)
            energy_cost = {
                'current': self.energy_cost_per_mwh,
                'projected': [self.energy_cost_per_mwh * (1 + self.energy_price_increase) ** (y - min(years)) for y in range(len(years))]
            }

            # Environmental damage cost (USD per ton of CO2)
            env_damage_cost = {
                'current': self.carbon_damage_cost,
                'projected': [self.carbon_damage_cost * (1 + self.carbon_damage_increase) ** (y - min(years)) for y in range(len(years))]
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

                # Add adoption-related data for Thailand
                if is_thailand and 'adoption_rates' in data:
                    benefits['economic'][scenario]['adoption_rates'] = data['adoption_rates']
                    benefits['economic'][scenario]['cumulative_adoption'] = data['cumulative_adoption']

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
            if is_thailand and sector_breakdown:
                carbon_credits[scenario].update({
                    'sector_breakdown': sector_breakdown
                })

        return carbon_credits
    def _create_visualizations(self, country, forecasts, results, benefits, carbon_credits):
        """
        Create visualizations for energy efficiency analysis with enhanced
        visualizations showing adoption curves and departures from BAU

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

        # 1. Energy Demand Scenarios - Departures from BAU
        plt.figure(figsize=(12, 8))

        # Plot BAU
        plt.plot(years, results['bau']['demand'], 'r-', linewidth=3, label='Business As Usual')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (scenario, data) in enumerate(results['scenarios'].items()):
            # Calculate reductions from BAU
            reductions = results['bau']['demand'] - data['demand']

            # Plot as filled area between BAU and scenario line
            plt.fill_between(years,
                             results['bau']['demand'] - reductions,
                             results['bau']['demand'],
                             color=colors[i], alpha=0.3,
                             label=f"{scenario.capitalize()} Reduction")

            # Plot the resulting demand line
            plt.plot(years, data['demand'], marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Scenario")

        plt.title(f"{country} - Energy Demand: Departures from BAU Trajectory", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Energy Demand (TWh)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_demand_scenarios.png'), dpi=300)
        plt.close()

        # 1b. Natural vs Efficiency-Driven Reductions (only for Thailand)
        if is_thailand and 'natural_efficiency' in results:
            plt.figure(figsize=(12, 8))

            # Plot BAU without any efficiency
            baseline_without_efficiency = results['bau']['demand'] / (1 - results['natural_efficiency'])
            plt.plot(years, baseline_without_efficiency, 'k-', linewidth=2, label='Baseline (No Efficiency)')

            # Plot BAU with natural efficiency improvements
            plt.plot(years, results['bau']['demand'], 'r-', linewidth=3, label='Business As Usual (With Natural Efficiency)')

            # Calculate natural efficiency savings
            natural_savings = baseline_without_efficiency - results['bau']['demand']

            # Plot natural efficiency as filled area
            plt.fill_between(years,
                             baseline_without_efficiency - natural_savings,
                             baseline_without_efficiency,
                             color='lightgray', alpha=0.5,
                             label=f"Natural Efficiency Improvements")

            # Plot scenario with most aggressive adoption
            best_scenario = max(results['scenarios'].keys(),
                                key=lambda s: results['scenarios'][s]['demand_reduction'][-1]
                                if 'demand_reduction' in results['scenarios'][s] else 0)

            best_data = results['scenarios'][best_scenario]
            plt.plot(years, best_data['demand'], linestyle='-', linewidth=2,
                     color='green', label=f"{best_scenario.capitalize()} Scenario")

            # Plot additional savings from efficiency program
            additional_savings = results['bau']['demand'] - best_data['demand']
            plt.fill_between(years,
                             results['bau']['demand'] - additional_savings,
                             results['bau']['demand'],
                             color='green', alpha=0.3,
                             label=f"Additional Program Savings")

            plt.title(f"{country} - Natural vs Program-Driven Efficiency Gains", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Energy Demand (TWh)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'natural_vs_program_efficiency.png'), dpi=300)
            plt.close()

        # 2. CO2 Emissions Scenarios - Departures from BAU
        plt.figure(figsize=(12, 8))

        # Plot BAU
        plt.plot(years, results['bau']['co2'], 'r-', linewidth=3, label='Business As Usual')

        for i, (scenario, data) in enumerate(results['scenarios'].items()):
            # Calculate reductions from BAU
            reductions = results['bau']['co2'] - data['co2']

            # Plot as filled area between BAU and scenario line
            plt.fill_between(years,
                             results['bau']['co2'] - reductions,
                             results['bau']['co2'],
                             color=colors[i], alpha=0.3,
                             label=f"{scenario.capitalize()} Reduction")

            # Plot the resulting emissions line
            plt.plot(years, data['co2'], marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Scenario")

        plt.title(f"{country} - CO₂ Emissions: Departures from BAU Trajectory", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("CO₂ Emissions (metric tons per capita)", fontsize=12)
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
            if is_thailand and carbon_credits and scenario in carbon_credits:
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
                # Original plot for non-Thailand countries or if carbon_credits is not available
                plt.bar(x + i * bar_width, data['net_benefit'] / 1e9, width=bar_width,
                        label=f"{scenario.capitalize()} Scenario", alpha=0.7)

        # Same for other parts of code that use carbon_credits
        # For example, in the Carbon Credits Value section:

        # 4. Carbon Credits Value
        if carbon_credits:  # Only create this plot if carbon_credits is valid
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

        # 6. NDC Contribution - BAU Departures
        plt.figure(figsize=(12, 8))

        # Estimate potential NDC contribution (example reduction targets)
        # Simplified assumption: 30% reduction from BAU by 2030
        bau_2030_index = np.where(years == 2030)[0][0] if 2030 in years else -1

        if bau_2030_index >= 0:
            bau_2030_co2 = results['bau']['co2'][bau_2030_index]
            ndc_target = bau_2030_co2 * 0.7  # 30% reduction

            # Plot BAU line
            plt.plot(years, results['bau']['co2'], 'r-', linewidth=3, label='Business As Usual')

            # Plot NDC target as horizontal line from 2030
            plt.plot([2030, years[-1]], [ndc_target, ndc_target],
                     'k--', linewidth=2, label='Example NDC Target (-30% from BAU)')

            # Plot departures from BAU for each scenario
            for i, (scenario, data) in enumerate(results['scenarios'].items()):
                # Calculate reduction from BAU
                reduction = results['bau']['co2'] - data['co2']

                # Plot as filled area between BAU and scenario line
                plt.fill_between(years,
                                 results['bau']['co2'] - reduction,
                                 results['bau']['co2'],
                                 color=colors[i], alpha=0.3,
                                 label=f"{scenario.capitalize()} Reduction")

                # Plot the scenario line
                plt.plot(years, data['co2'], marker='o', linestyle='-', linewidth=2,
                         color=colors[i], label=f"{scenario.capitalize()} Scenario")

                # Add NDC contribution annotations
                scenario_2030_co2 = data['co2'][bau_2030_index]
                reduction_pct = (bau_2030_co2 - scenario_2030_co2) / bau_2030_co2 * 100
                ndc_contribution = (bau_2030_co2 - scenario_2030_co2) / (bau_2030_co2 - ndc_target) * 100

                plt.annotate(
                    f"{scenario.capitalize()}: {reduction_pct:.1f}% reduction\n({ndc_contribution:.1f}% of NDC target)",
                    xy=(2030, scenario_2030_co2), xytext=(2031, scenario_2030_co2),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colors[i]),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8),
                    fontsize=10)

        plt.title(f"{country} - Contribution to NDC Targets", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("CO₂ Emissions (metric tons per capita)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ndc_contribution.png'), dpi=300)
        plt.close()

        # 7. Yearly Energy Consumption Reduction (vs fixed)
        plt.figure(figsize=(12, 8))

        for i, (scenario, data) in enumerate(results['scenarios'].items()):
            # Use demand_reduction if available (for Thailand), otherwise calculate
            if 'demand_reduction' in data:
                yearly_reduction = data['demand_reduction'] * 100  # Convert to percentage
            else:
                yearly_reduction = (results['bau']['demand'] - data['demand']) / results['bau']['demand'] * 100

            plt.plot(years, yearly_reduction, marker='o', linestyle='-', linewidth=2,
                     color=colors[i], label=f"{scenario.capitalize()} Scenario")

        # Plot natural efficiency improvement if available
        if 'natural_efficiency' in results:
            plt.plot(years, results['natural_efficiency'] * 100, 'k--', linewidth=2,
                     label='Natural Efficiency Improvement')

        plt.title(f"{country} - Yearly Energy Consumption Reduction", fontsize=15)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Reduction from BAU (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'yearly_reduction.png'), dpi=300)
        plt.close()

        # Thailand-specific visualizations
        if is_thailand:
            # 8. Adoption Curves
            plt.figure(figsize=(12, 8))

            for i, (scenario, data) in enumerate(results['scenarios'].items()):
                if 'cumulative_adoption' in data:
                    plt.plot(years, data['cumulative_adoption'] * 100, marker='o', linestyle='-',
                             linewidth=2, color=colors[i], label=f"{scenario.capitalize()} Adoption")
                elif 'adoption_rates' in benefits['economic'].get(scenario, {}):
                    # Try to get from benefits if not in results
                    plt.plot(years, benefits['economic'][scenario]['cumulative_adoption'] * 100, marker='o',
                             linestyle='-', linewidth=2, color=colors[i],
                             label=f"{scenario.capitalize()} Adoption")

            plt.title(f"{country} - Energy Efficiency Technology Adoption Curves", fontsize=15)
            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Cumulative Adoption (%)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'adoption_curves.png'), dpi=300)
            plt.close()

            # 9. Appliance Turnover Impact
            if 'natural_efficiency' in results:
                plt.figure(figsize=(12, 8))

                # Plot natural turnover efficiency improvement
                plt.plot(years, results['natural_efficiency'] * 100, 'k-', linewidth=3,
                         label='Natural Turnover Efficiency')

                # Plot accelerated turnover for different scenarios
                for i, (scenario, data) in enumerate(results['scenarios'].items()):
                    if 'params' in data and 'appliance_turnover_factor' in data['params']:
                        turnover_factor = data['params']['appliance_turnover_factor']
                        if turnover_factor > 1.0:
                            # Calculate accelerated efficiency
                            accelerated_efficiency = 1 - (1 - results['natural_efficiency']) * turnover_factor
                            plt.plot(years, accelerated_efficiency * 100, linestyle='--', linewidth=2,
                                     color=colors[i], label=f"{scenario.capitalize()} Accelerated Turnover")

                plt.title(f"{country} - Impact of Appliance Turnover on Efficiency", fontsize=15)
                plt.xlabel("Year", fontsize=12)
                plt.ylabel("Efficiency Improvement (%)", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'appliance_turnover.png'), dpi=300)
                plt.close()

            # 10. Historical Energy Efficiency Investments
            if hasattr(self, 'thailand_ee_data') and 'historical_investments' in self.thailand_ee_data:
                plt.figure(figsize=(12, 6))

                # Plot historical data
                historical = self.thailand_ee_data['historical_investments']
                plt.bar(historical['Year'], historical['Investment_Million_USD'], color='skyblue', alpha=0.7,
                        label='Historical Investments')

                # Plot trend line
                z = np.polyfit(historical['Year'], historical['Investment_Million_USD'], 1)
                p = np.poly1d(z)
                plt.plot(historical['Year'], p(historical['Year']), 'r--', label='Trend')

                # Add future projection if we have implementation costs
                if len(results['scenarios']) > 0:
                    # Take medium scenario if available, otherwise first scenario
                    scenario_key = 'medium' if 'medium' in results['scenarios'] else list(results['scenarios'].keys())[0]
                    if 'implementation_cost' in results['scenarios'][scenario_key]:
                        future_years = results['years']
                        future_investments = results['scenarios'][scenario_key]['implementation_cost'] / 1e6  # to millions
                        plt.plot(future_years, future_investments, 'g-', marker='o',
                                 label=f"Projected ({scenario_key.capitalize()} Scenario)")

                plt.title(f"{country} - Historical & Projected Energy Efficiency Investments", fontsize=15)
                plt.xlabel("Year", fontsize=12)
                plt.ylabel("Million USD", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'historical_investments.png'), dpi=300)
                plt.close()

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

            # Generate visualizations
            self._create_visualizations(country, forecasts, results, benefits, carbon_credits)

            print(f"\nCompleted energy efficiency analysis for {country}")
            print(f"Check output/efficiency_analysis/{country} for results and visualizations")

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

import os
import argparse
from asean_forecaster import ASEANForecaster
from energy_efficiency_analyzer import run_efficiency_analysis

def main():
    """Main entry point for ASEAN forecasting and energy efficiency analysis"""
    parser = argparse.ArgumentParser(description='ASEAN Forecaster with Energy Efficiency Analysis')
    parser.add_argument('--method', type=str, choices=['var', 'prophet', 'both'], default='both',
                        help='Forecasting method to use (default: both)')
    parser.add_argument('--file', type=str, default='ASEAN_2035.csv',
                        help='Input CSV file (default: ASEAN_2035.csv)')
    parser.add_argument('--efficiency', action='store_true',
                        help='Run energy efficiency analysis after forecasting')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Initialize forecaster without selecting a country yet
    print(f"Loading data from {args.file}...")
    temp_forecaster = ASEANForecaster(args.file)

    # Get list of available countries
    available_countries = temp_forecaster.data['Country'].unique()
    available_countries = [str(country) for country in available_countries]

    # Display available countries
    print("\nAvailable countries:")
    for i, country in enumerate(available_countries, 1):
        print(f"{i}. {country}")

    # Ask user to select a country
    while True:
        try:
            selection = input("\nSelect a country (enter number or name) or 'all' for all countries: ")

            # Check if user wants all countries
            if selection.lower() == 'all':
                selected_country = None
                print("Analyzing all countries")
                break

            # Check if selection is a number
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(available_countries):
                    selected_country = available_countries[index]
                    print(f"Selected: {selected_country}")
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(available_countries)}")
            # Check if selection is a country name
            elif selection in available_countries:
                selected_country = selection
                print(f"Selected: {selected_country}")
                break
            else:
                print("Invalid selection. Please try again.")
        except Exception as e:
            print(f"Error: {e}")

    # Ask if user wants to skip forecasting
    skip_forecast = False
    while True:
        skip_response = input("\nDo you want to skip forecasting and use existing forecast files? (y/n): ")
        if skip_response.lower() in ['y', 'yes']:
            skip_forecast = True
            break
        elif skip_response.lower() in ['n', 'no']:
            skip_forecast = False
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

    # Re-initialize forecaster with selected country
    forecaster = ASEANForecaster(args.file, selected_country)

    # Run forecasts if not skipped
    if not skip_forecast:
        print(f"\nStarting ASEAN Forecaster with parameters:")
        print(f"  Country: {selected_country if selected_country else 'All countries'}")
        print(f"  Method: {args.method}")
        print(f"  File: {args.file}")

        # Plot correlation matrix
        forecaster.plot_correlation_matrix()

        # Run forecasts
        if args.method in ['var', 'both']:
            forecaster.run_var_forecasts()

        if args.method in ['prophet', 'both']:
            forecaster.run_prophet_forecasts()

        # Combine forecasts
        if args.method == 'both':
            forecaster.combine_forecasts()

        print("\nForecasting complete!")
    else:
        print("\nSkipping forecasting as requested...")

    # Run or ask about energy efficiency analysis
    if args.efficiency:
        # Automatically run efficiency analysis if flag was provided
        run_efficiency_analysis(forecaster)
    else:
        # Ask if user wants to run energy efficiency analysis
        while True:
            response = input("\nWould you like to run energy efficiency analysis? (y/n): ")
            if response.lower() in ['y', 'yes']:
                run_efficiency_analysis(forecaster)
                break
            elif response.lower() in ['n', 'no']:
                print("Skipping energy efficiency analysis.")
                break
            else:
                print("Invalid response. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()

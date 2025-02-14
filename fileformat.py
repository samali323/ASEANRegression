import pandas as pd
from datetime import datetime
import os
import glob

def get_latest_wb_file():
    """Find the most recent World Bank data file"""
    pattern = 'worldbank_data/ASEAN_combined_*.csv'
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No World Bank data files found in worldbank_data directory")
    return max(files, key=os.path.getctime)

def merge_datasets():
    """
    Merge ASEANIndicators with World Bank data
    """
    try:
        # Read existing ASEAN data
        asean_df = pd.read_csv(r'C:\Users\samal\Documents\ASEANIndicators.csv')
        print("Loaded ASEAN Indicators data")

        # Get latest World Bank data file
        wb_file = get_latest_wb_file()
        print(f"Found World Bank data file: {wb_file}")

        # Read World Bank data
        wb_df = pd.read_csv(wb_file)
        print("Loaded World Bank data")

        # Show initial data info
        print("\nInitial Data Summary:")
        print(f"ASEAN data shape: {asean_df.shape}")
        print(f"World Bank data shape: {wb_df.shape}")

        # Merge datasets
        merged_df = pd.merge(asean_df, wb_df,
                             on=['Country', 'Year'],
                             how='left')

        # Remove duplicate columns if any
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # Save merged dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'ASEANIndicators_Enhanced_{timestamp}.csv'
        merged_df.to_csv(output_file, index=False)

        print(f"\nData Merge Summary:")
        print("-" * 20)
        print(f"Original ASEAN indicators: {len(asean_df.columns)}")
        print(f"World Bank indicators added: {len(wb_df.columns)-2}")  # -2 for Country and Year
        print(f"Total indicators in merged dataset: {len(merged_df.columns)}")
        print(f"\nSaved to: {output_file}")

        return merged_df

    except Exception as e:
        print(f"Error during merge: {str(e)}")
        raise

def analyze_merged_data(merged_df):
    """
    Analyze the completeness of the merged dataset
    """
    print("\nData Completeness Analysis:")
    print("-" * 20)

    # Calculate completeness by indicator
    completeness = (merged_df.count() / len(merged_df) * 100).sort_values()

    print("\nIndicators with lowest completeness:")
    print(completeness.head().to_string())
    print("\nIndicators with highest completeness:")
    print(completeness.tail().to_string())

    # Print detailed information about each column
    print("\nDetailed Column Information:")
    print("-" * 20)
    for col in merged_df.columns:
        missing = merged_df[col].isnull().sum()
        pct_missing = (missing / len(merged_df)) * 100
        print(f"\n{col}:")
        print(f"Missing values: {missing} ({pct_missing:.1f}%)")
        if merged_df[col].dtype in ['float64', 'int64']:
            print(f"Range: {merged_df[col].min():.2f} to {merged_df[col].max():.2f}")

    return completeness

if __name__ == "__main__":
    try:
        # Merge datasets
        print("Starting data merge process...")
        merged_df = merge_datasets()

        # Analyze merged data
        print("\nAnalyzing merged data...")
        completeness = analyze_merged_data(merged_df)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

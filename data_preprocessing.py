import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the preprocessor with the input file

        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing the data
        """
        self.original_df = pd.read_csv(file_path)
        self.processed_df = self.original_df.copy()

    def handle_missing_values(self, method='advanced'):
        """
        Handle missing values using different strategies

        Parameters:
        -----------
        method : str, optional (default='advanced')
            'simple': Use mean imputation
            'knn': Use KNN imputer
            'advanced': Use multiple interpolation techniques

        Returns:
        --------
        pandas.DataFrame
            DataFrame with handled missing values
        """
        # Create a copy of the dataframe
        df = self.processed_df.copy()

        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if method == 'simple':
            # Simple mean imputation
            imputer = SimpleImputer(strategy='mean')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        elif method == 'knn':
            # KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        elif method == 'advanced':
            # Advanced interpolation
            for col in numeric_columns:
                # Try different interpolation methods
                interpolation_methods = [
                    lambda s: s.interpolate(method='linear', limit_direction='both'),
                    lambda s: s.interpolate(method='quadratic', limit_direction='both'),
                    lambda s: s.interpolate(method='cubic', limit_direction='both'),
                    lambda s: s.fillna(s.mean()),
                    lambda s: s.fillna(method='ffill'),
                    lambda s: s.fillna(method='bfill')
                ]

                for method in interpolation_methods:
                    try:
                        interpolated = method(df[col].copy())
                        if not interpolated.isnull().any():
                            df[col] = interpolated
                            break
                    except Exception as e:
                        print(f"Interpolation method failed for {col}: {e}")

        self.processed_df = df
        return df

    def create_polynomial_features(self, columns=None):
        """
        Create polynomial features for specified columns

        Parameters:
        -----------
        columns : list, optional
            Columns to create polynomial features for
            If None, uses ['Population', 'GDP(USD)']

        Returns:
        --------
        pandas.DataFrame
            DataFrame with added polynomial features
        """
        if columns is None:
            columns = ['Population', 'GDP(USD)']

        df = self.processed_df.copy()

        for col in columns:
            df[f'{col}_Squared'] = df[col] ** 2
            df[f'{col}_Cubed'] = df[col] ** 3

        self.processed_df = df
        return df

    def standardize_features(self, columns=None):
        """
        Standardize specified features

        Parameters:
        -----------
        columns : list, optional
            Columns to standardize
            If None, uses all numeric columns

        Returns:
        --------
        pandas.DataFrame
            DataFrame with standardized features
        """
        df = self.processed_df.copy()

        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        # Create scaler
        scaler = StandardScaler()

        # Standardize specified columns
        df[columns] = scaler.fit_transform(df[columns])

        self.processed_df = df
        return df

    def save_processed_data(self, output_path=None):
        """
        Save processed data to a CSV file

        Parameters:
        -----------
        output_path : str, optional
            Path to save the processed CSV
            If None, uses original filename with '_processed' suffix
        """
        if output_path is None:
            # Create output filename
            base_path = self.original_df.attrs.get('file_path', 'processed_data')
            output_path = base_path.replace('.csv', '_processed.csv')

        # Save processed dataframe
        self.processed_df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

        return output_path

    def generate_data_report(self):
        """
        Generate a comprehensive report on data preprocessing

        Returns:
        --------
        dict
            Report with various data statistics
        """
        report = {
            'original_shape': self.original_df.shape,
            'processed_shape': self.processed_df.shape,
            'missing_values_original': self.original_df.isnull().sum(),
            'missing_values_processed': self.processed_df.isnull().sum(),
            'columns': list(self.processed_df.columns)
        }

        return report

def main():
    # Find the most recent enhanced dataset
    import glob
    import os

    # Find all files matching the pattern
    enhanced_files = glob.glob('*ASEANIndicators_Enhanced*.csv')

    if not enhanced_files:
        print("No enhanced dataset found. Please check your file path.")
        exit()

    # Get the most recent file
    latest_file = max(enhanced_files, key=os.path.getctime)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(latest_file)

    # Apply preprocessing steps
    preprocessor.handle_missing_values(method='advanced')
    preprocessor.create_polynomial_features()
    preprocessor.standardize_features()

    # Generate and print report
    report = preprocessor.generate_data_report()
    print("\nData Preprocessing Report:")
    for key, value in report.items():
        print(f"{key}: {value}")

    # Save processed data
    processed_file = preprocessor.save_processed_data()

    return processed_file

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def calculate_adjusted_r_squared(y_true, y_pred, num_features):
    """
    Calculate adjusted R-squared

    Parameters:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    num_features (int): Number of features in the model

    Returns:
    float: Adjusted R-squared value
    """
    n = len(y_true)  # number of observations
    p = num_features  # number of predictors

    # Calculate R-squared
    r_squared = r2_score(y_true, y_pred)

    # Calculate adjusted R-squared
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    return adjusted_r_squared

def analyze_indicators(file_path):
    """
    Analyze the importance of different indicators for electricity demand
    using both linear regression and Lasso regression
    """
    # Read data
    df = pd.read_csv(file_path)

    # Create squared terms for Population and GDP
    df['Population_Squared'] = df['Population'] ** 2
    df['GDP(USD)_Squared'] = df['GDP(USD)'] ** 2

    # Prepare features
    features = [
        'Population', 'Population_Squared',
        'GDP(USD)', 'GDP(USD)_Squared',
        'Industry (% of GDP)',
        'Manufacturing (% of GDP)',
        'Foreign direct investment (% of GDP)',
        'Energy use (kg of oil equivalent per capita)',
        'Electric power consumption (kWh per capita)',
        'Access to electricity (% of population)',
        'Electric power transmission and distribution losses (% of output)',
        'Fossil fuel energy consumption (% of total)',
        'Urban population (% of total)',
        'Individuals using the Internet (% of population)',
        'Mobile cellular subscriptions (per 100 people)',
        'PM2.5 air pollution (micrograms per cubic meter)'
    ]

    # Drop rows with missing values
    model_df = df[['Country', 'Year', 'Demand (TWh)'] + features].dropna()

    # Prepare X and y
    X = model_df[features]
    y = model_df['Demand (TWh)']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Fit Linear Regression
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    lr_pred = lr.predict(X_scaled)

    # Get Linear Regression coefficients
    lr_coef_raw = lr.coef_
    lr_coef = pd.DataFrame({
        'Feature': features,
        'Coefficient (Raw)': lr_coef_raw,
        'Coefficient (Formatted)': [f'{coef:.2f}' for coef in lr_coef_raw],
        'Absolute Coefficient': np.abs(lr_coef_raw)
    }).sort_values('Absolute Coefficient', ascending=False)

    # Fit Lasso with cross-validation
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_pred = lasso.predict(X_scaled)

    # Get Lasso coefficients
    lasso_coef_raw = lasso.coef_
    lasso_coef = pd.DataFrame({
        'Feature': features,
        'Coefficient (Raw)': lasso_coef_raw,
        'Coefficient (Formatted)': [f'{coef:.2f}' for coef in lasso_coef_raw],
        'Absolute Coefficient': np.abs(lasso_coef_raw)
    }).sort_values('Absolute Coefficient', ascending=False)

    # Calculate R-squared and Adjusted R-squared for both models
    lr_r2 = r2_score(y, lr_pred)
    lr_adjusted_r2 = calculate_adjusted_r_squared(y, lr_pred, len(features))

    lasso_r2 = r2_score(y, lasso_pred)
    lasso_num_features = len([f for f in lasso.coef_ if f != 0])
    lasso_adjusted_r2 = calculate_adjusted_r_squared(y, lasso_pred, lasso_num_features)

    # Print results in tabular format
    print("\nLINEAR REGRESSION RESULTS")
    print("=" * 50)
    print(f"R-squared: {lr_r2:.2f}")
    print(f"Adjusted R-squared: {lr_adjusted_r2:.2f}")
    print("\nFeature Importance (Highest to Lowest):")

    # Create a display DataFrame with only formatted coefficients for display
    lr_coef_display = lr_coef[['Feature', 'Coefficient (Formatted)']]
    print(tabulate(lr_coef_display, headers='keys', tablefmt='pretty', showindex=False))

    print("\nLASSO REGRESSION RESULTS")
    print("=" * 50)
    print(f"R-squared: {lasso_r2:.2f}")
    print(f"Adjusted R-squared: {lasso_adjusted_r2:.2f}")
    print(f"Alpha selected: {lasso.alpha_:.2f}")
    print(f"Number of features used: {lasso_num_features}")
    print("\nFeature Importance (Highest to Lowest):")

    # Create a display DataFrame with only formatted coefficients for display
    lasso_coef_display = lasso_coef[['Feature', 'Coefficient (Formatted)']]
    print(tabulate(lasso_coef_display, headers='keys', tablefmt='pretty', showindex=False))

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Plot Linear Regression coefficients
    plt.subplot(1, 2, 1)
    sns.barplot(x='Coefficient (Raw)', y='Feature', data=lr_coef)
    plt.title('Features (Linear Regression)')
    plt.xlabel('Standardized Coefficient')

    # Plot Lasso coefficients
    plt.subplot(1, 2, 2)
    lasso_nonzero = lasso_coef[lasso_coef['Coefficient (Raw)'] != 0]
    sns.barplot(x='Coefficient (Raw)', y='Feature', data=lasso_nonzero)
    plt.title('Non-Zero Features (Lasso)')
    plt.xlabel('Standardized Coefficient')

    plt.tight_layout()
    plt.show()

    # Return coefficients for further analysis
    return lr_coef, lasso_coef, lr_r2, lasso_r2, lr_adjusted_r2, lasso_adjusted_r2

def plot_relationship(df, x_col, y_col='Demand (TWh)'):
    """Plot relationship between a feature and electricity demand"""
    plt.figure(figsize=(10, 6))

    # Create scatter plot with different color for each country
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country]
        plt.scatter(country_data[x_col], country_data[y_col],
                    label=country, alpha=0.6)

    # Add trend line
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Relationship between {x_col} and {y_col}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Add this at the bottom to actually run the analysis
if __name__ == "__main__":
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

    print(f"Analyzing file: {latest_file}")

    # Run the analysis
    lr_coef, lasso_coef, lr_r2, lasso_r2, lr_adjusted_r2, lasso_adjusted_r2 = analyze_indicators(latest_file)

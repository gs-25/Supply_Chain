import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Read the dataset
df = pd.read_csv('Supermart Grocery Sales - Retail Analytics Dataset.csv')

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')

# Function to determine season based on month
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

# Add season column
df['Season'] = df['Order Date'].apply(get_season)

# Function to analyze seasonal patterns
def analyze_seasonal_patterns(data):
    # Calculate seasonal metrics
    seasonal_analysis = data.groupby(['Category', 'Season'])['Sales'].agg([
        ('Total Sales', 'sum'),
        ('Average Daily Sales', 'mean'),
        ('Number of Orders', 'count')
    ]).reset_index()
    
    # Calculate percentage of total sales for each category in each season
    for category in seasonal_analysis['Category'].unique():
        category_total = seasonal_analysis[seasonal_analysis['Category'] == category]['Total Sales'].sum()
        mask = seasonal_analysis['Category'] == category
        seasonal_analysis.loc[mask, 'Sales Percentage'] = (
            seasonal_analysis.loc[mask, 'Total Sales'] / category_total * 100
        ).round(2)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    pivot_data = seasonal_analysis.pivot(
        index='Category', 
        columns='Season', 
        values='Sales Percentage'
    )
    
    # Sort categories by total sales
    category_totals = data.groupby('Category')['Sales'].sum()
    pivot_data = pivot_data.reindex(category_totals.sort_values(ascending=False).index)
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        center=25,
        cbar_kws={'label': '% of Category Sales'}
    )
    
    plt.title('Seasonal Sales Distribution by Category', pad=20)
    plt.tight_layout()
    save_plot(plt.gcf(), 'seasonal_analysis.png')
    
    # Create bar plot of average daily sales by season
    plt.figure(figsize=(10, 6))
    season_order = ['Summer', 'Monsoon', 'Post-Monsoon', 'Winter']
    avg_sales = data.groupby('Season')['Sales'].mean().reindex(season_order)
    
    sns.barplot(x=avg_sales.index, y=avg_sales.values, palette='viridis')
    plt.title('Average Daily Sales by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Daily Sales (₹)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(plt.gcf(), 'seasonal_averages.png')
    
    return seasonal_analysis

# Function to perform time series analysis
def analyze_time_series(data, category):
    # Aggregate data by month
    monthly_data = data[data['Category'] == category].set_index('Order Date')
    monthly_data = monthly_data['Sales'].resample('M').sum()
    
    # Fill missing months with forward fill
    monthly_data = monthly_data.asfreq('M', method='ffill')
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Monthly Sales
    ax1.plot(monthly_data.index, monthly_data.values, marker='o')
    ax1.set_title(f'Monthly Sales - {category}')
    ax1.set_xlabel('')
    ax1.set_ylabel('Total Sales (₹)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Year-over-Year Comparison
    years = monthly_data.index.year.unique()
    for year in years:
        year_data = monthly_data[monthly_data.index.year == year]
        ax2.plot(range(1, 13), year_data.values[:12], 
                marker='o', label=str(year))
    
    ax2.set_title('Year-over-Year Comparison')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total Sales (₹)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Seasonal Pattern
    seasonal_avg = monthly_data.groupby(monthly_data.index.month).mean()
    ax3.plot(range(1, 13), seasonal_avg.values, marker='o', color='green')
    ax3.set_title('Average Seasonal Pattern')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Sales (₹)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(plt.gcf(), f'time_series_{category}.png')
    
    return monthly_data

# Function to build and evaluate SARIMA model
def build_sarima_model(data, category):
    # Aggregate data by month
    monthly_data = data[data['Category'] == category].set_index('Order Date')
    monthly_data = monthly_data['Sales'].resample('M').sum()
    monthly_data = monthly_data.asfreq('M', method='ffill')
    
    # Split data into train and test sets
    train_size = int(len(monthly_data) * 0.8)
    train_data = monthly_data[:train_size]
    test_data = monthly_data[train_size:]
    
    # Build SARIMA model
    model = SARIMAX(train_data,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    
    # Make predictions
    forecast = results.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, forecast)
    mae = mean_absolute_error(test_data, forecast)
    
    # Plot results
    plt.figure(figsize=(15, 8))
    plt.plot(train_data.index, train_data, 
             label='Historical Data', color='blue', alpha=0.6)
    plt.plot(test_data.index, test_data, 
             label='Actual', color='green', alpha=0.6)
    plt.plot(test_data.index, forecast, 
             label='Forecast', color='red', linewidth=2)
    
    plt.title(f'Sales Forecast - {category}')
    plt.xlabel('Date')
    plt.ylabel('Monthly Sales (₹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), f'forecast_{category}.png')
    
    return mse, mae, results

# Main analysis
def main():
    print("Performing Seasonal Analysis...")
    seasonal_analysis = analyze_seasonal_patterns(df)
    
    # Get unique categories
    categories = df['Category'].unique()
    
    # Perform analysis for each category
    print("\nPerforming Time Series Analysis...")
    results = {}
    for category in categories:
        print(f"\nAnalyzing {category}...")
        
        # Perform time series analysis
        monthly_data = analyze_time_series(df, category)
        
        # Build and evaluate SARIMA model
        mse, mae, model = build_sarima_model(df, category)
        
        results[category] = {
            'MSE': mse,
            'MAE': mae,
            'Model': model
        }
        
        print(f"Results for {category}:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")

# Function to save plots with consistent styling
def save_plot(fig, filename):
    """Save plot to output directory with consistent styling"""
    output_path = os.path.join('output', filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main() 
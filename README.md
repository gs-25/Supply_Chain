# Supply Chain Demand Forecasting Analysis

This project implements advanced time series analysis techniques for supply chain demand forecasting, focusing on seasonal patterns and trend analysis.

## Project Overview

The analysis utilizes the Supermart Grocery Sales Retail Analytics Dataset to:
- Analyze seasonal patterns in sales data
- Implement time series forecasting using SARIMA models
- Generate visualizations for trend analysis
- Provide actionable insights for inventory management

## Features

- Seasonal analysis with Indian climate-based season definitions
- Time series analysis with monthly aggregation
- SARIMA-based demand forecasting
- Comprehensive visualization suite
- Detailed LaTeX report generation

## Requirements

- Python 3.x
- Required Python packages (see requirements.txt)
- LaTeX distribution (for report generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gs-25/Supply_Chain.git
cd Supply_Chain
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the analysis:
```bash
python demand_forecasting.py
```

2. Generate the report:
```bash
pdflatex report.tex
```

## Project Structure

- `demand_forecasting.py`: Main analysis script
- `report.tex`: LaTeX source for the report
- `output/`: Directory containing generated visualizations
- `requirements.txt`: Python package dependencies

## Author

GURJOT SINGH BAJAJ
Roll Number: 22124040

## Dataset

The analysis uses the Supermart Grocery Sales Retail Analytics Dataset from Kaggle:
[Dataset Link](https://www.kaggle.com/datasets/mohamedharris/supermart-grocery-sales-retail-analytics-dataset?resource=download) 
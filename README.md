# Movie Revenue Analysis

## Team Members
- Mohammad Wael (UFID: 9470 7697)
- Vishal Karthikeyan Setti (UFID: 4767 0880)

## Project Overview
This project performs a detailed exploratory data analysis (EDA) on a movie dataset to uncover revenue trends by year and month. The objective is to understand historical revenue shifts, identify seasonal impacts on revenue, and predict revenue which can be used for optimal movie release periods.

## Project Structure
```
cap5771sp25-project/
│
├── .idea                           # PyCharm IDE configuration files
├── data                            # Raw datasets storage
│   └── data.txt                    # Data documentation file
├── features                        # Processed feature outputs
│   ├── features_scores.txt         # Feature importance metrics
│   └── significant_features.txt    # Statistically significant features
├── Report                          # Project documentation
│   ├── Milestone1.pdf              # Milestone 1 report
│   └── Milestone2.pdf              # Milestone 2 report
├── Scripts                         # Source code directory
│   ├── __pycache__                # Python bytecode cache
│   ├── main.ipynb                  # Jupyter notebook for Milestone 1 and Milestone 2
│   ├── preprocessor.py             # Data cleaning scripts
│   └── requirements.txt            # Python dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project overview document (This file)
```

## Dataset Description
The dataset used in this analysis contains various attributes of movies, including release dates, genres, budgets, revenues, and audience and critic ratings. The data covers films released over the past century, providing a comprehensive view of the industry dynamics.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.x
- Jupyter Notebook or any Python IDE

### Required Datasets
- [IMDb](https://datasets.imdbws.com/) 
- [TMDb](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
- [Rotten Tomatoes](https://www.kaggle.com/datasets/asaniczka/rotten-tomatoes-movies-and-critics-dataset)
- [COUNTRY CODES ALPHA-2 & ALPHA-3](https://www.kaggle.com/datasets/emolodov/country-codes-alpha2-alpha3/data)

### Required Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- sklearn
- plotly

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scipy sklearn plotly
```
## Getting Started
Clone the repository to your local machine:
```bash
git clone https://github.com/itswael/cap5771sp25-project.git
```
download the datasets and place them in the 'Data' folder:
```bash
cap5771sp25-project/Data
```
Navigate to the cloned repository:
```bash
cd downloadedRepositoryPath/cap5771sp25-project
```
Navigate to the Scripts folder:
```bash
cd Scripts
```
Open the Jupyter Notebook:
```bash
jupyter notebook
```
Update the file path in the notebook to point to the correct dataset location.

Open movie analysis.ipynb and run the cells sequentially to reproduce the analysis.

# Contributions
- Mohammad Wael: 
  - Data Cleaning, Data Analysis, Data Visualization, and Report Writing
  - on the IMDb title.basics, tmdb, rotten tomatoes critic and rotten tomatoes movies datasets
- Vishal Karthikeyan Setti: 
  - Data Cleaning, Data Analysis, Data Visualization, and Report Writing
  - on the IMDb title.ratings, title.principals, title.akas, title.crew and country codes datasets
- Both team members contributed equally to the project.
- [asaniczka](https://www.kaggle.com/asaniczka) for providing the TMDb datasets used in this analysis.
- [Emil Molodov](https://www.kaggle.com/emolodov) for providing the country codes dataset.
- [stefano leone](https://www.kaggle.com/stefanoleone992) for providing the Rotten Tomatoes datasets used in this analysis.
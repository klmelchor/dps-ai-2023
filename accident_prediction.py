import pandas as pd
import numpy as np
import os

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


# function used to create the needed cleaned datasets for each category (called by data_initialization)
def create_datasets(df, acc='all'):
    acc_to_use = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle'] if acc == 'all' else [acc]
    df_subset = df[(df['MONATSZAHL'].isin(acc_to_use)) & (df['JAHR'] <= 2020) & (df['MONAT'] != 'Summe') 
                & (df['AUSPRÄGUNG'] == 'insgesamt')].iloc[:,2:6].fillna(0)
    df_subset = df_subset.rename(columns={'JAHR': 'Year', 'MONAT': 'Month', 'WERT': 'Total', 'VORJAHRESWERT': 'Prev_Year'})

    # we use the Month_Clean column to process the test parameter
    month_list = list(df_subset['Month'].str[-2:])
    month_list_clean = [int(x) if int(x) >= 10 else int(x[-1:]) for x in month_list]
    df_subset['Month_Clean'] = month_list_clean

    df_subset.to_pickle(str(acc) + ".pkl")
    return df_subset

# function to initialize the datasets
def data_initialization(df):
    accident_cats = df['MONATSZAHL'].unique()
    for acc in accident_cats:
        file_path = str(acc) + '.pkl'
        if os.path.isfile(file_path) == True:
            print(file_path, 'exists, skipping...')
            continue
        else:
            create_datasets(acc)

# function to normalize input
def normalize_values(year_val, month_val, df_to_use):
    month_val_conv = int(str(year_val) + str(month_val) if month_val >= 10 else str(year_val) + '0' + str(month_val))
    
    # find the value for the previous year, since this is one of the features needed by LinRes
    prev_year_val = int(df_to_use[(df_to_use['Year'] == year_val - 1) & 
                                    (df_to_use['Month_Clean'] == month_val)]['Total'])

    year_val_norm = (year_val - df_to_use['Year'].mean()) / df_to_use['Year'].std()
    prev_year_val_norm = (prev_year_val - df_to_use['Prev_Year'].mean()) / df_to_use['Prev_Year'].std()
    month_val_conv_norm = (month_val_conv - pd.to_numeric(df_to_use['Month']).mean()) / pd.to_numeric(df_to_use['Month']).std()
    
    return year_val_norm, prev_year_val_norm, month_val_conv_norm

# training using SGDR
def train(df_to_use):
    X_features = ['Year', 'Month', 'Prev_year']
    X_train = df_to_use.iloc[:, [0,1,3]]
    y_train = df_to_use['Total']

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train)

    sgdr = SGDRegressor(max_iter=1000)
    sgdr.fit(X_norm, y_train)
    print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

    b_norm = sgdr.intercept_
    w_norm = sgdr.coef_
    print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
    
    return sgdr, w_norm, b_norm

def main_func(year, month):
    df = pd.read_csv('monatszahlen2209_verkehrsunfaelle.csv')
    
    # initialize the datasets for alcohol, escape, and traffic
    data_initialization(df)

    # using alcohol for now, this can be changed to use all catergories
    df_alcohol = pd.read_pickle('Alkoholunfälle.pkl')

    # normalization of input since the value ranges of the features are far apart and the model uses normalized values
    year_val_norm, prev_year_val_norm, month_val_conv_norm = normalize_values(year, month, df_alcohol)
    
    # train and get model
    sgdr, w_norm, b_norm = train(df_alcohol)

    # prediction and return it as float
    y_pred_sgd = sgdr.predict([[year_val_norm, month_val_conv_norm, prev_year_val_norm]])  
    return float(y_pred_sgd[0])
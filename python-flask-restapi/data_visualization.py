import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_totals(acc):
    sns.set(style="ticks")
    figure(figsize=(10, 6), dpi=100)
    
    acc_subset = df[(df['MONATSZAHL'] == acc) & (df['MONAT'] == 'Summe') & (df['AUSPRÄGUNG'] == 'insgesamt')].reset_index()
    
    l_plot = sns.lineplot(data=acc_subset, x='JAHR', y='WERT', marker='*', linewidth=2, markersize=10)

    plot_name = 'Total Number of ' + acc + ' Accidents per Year'
    l_plot.set(title=plot_name)
    
    for x, y in zip(acc_subset['JAHR'], acc_subset['WERT']):
        plt.text(x = x+0.5, y = y-10, s = '{:.0f}'.format(y), color='black', fontsize=10)

    plt.xticks(np.arange(2000, 2022, 1.0), fontsize=10, rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Number of Accidents')
    plt.grid()
    
    filename = acc + '_Accidents_per_Year.png'
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    # (2086, 9)
    df = pd.read_csv('monatszahlen2209_verkehrsunfaelle.csv')
    df_colnames = list(df.columns)

    # find all unique category values
    accident_cats = df[df_colnames[0]].unique() 

    # values are ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']

    for acc in accident_cats:
        plot_totals(acc)



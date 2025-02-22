import pandas as pd

csv_path = 'data/leads_data.csv'

#Daten einlesen
df = pd.read_csv(csv_path)

print('HEAD:')
print(df.head())
print('info:')
print(df.info())
print('describe:')
print(df.describe())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import csv
from py2neo import Graph, Node, Relationship

csv_path = 'data/leads_data.csv'

# Daten einlesen
leads_df = pd.read_csv(csv_path)

print('HEAD:')
print(leads_df.head())
print('info:')
print(leads_df.info())
print('describe:')
print(leads_df.describe())

# Fehlende Werte behandeln
leads_df['Country'] = leads_df['Country'].fillna('Unknown')
leads_df['TotalVisits'] = leads_df['TotalVisits'].fillna(leads_df['TotalVisits'].median())
leads_df['Page Views Per Visit'] = leads_df['Page Views Per Visit'].fillna(leads_df['Page Views Per Visit'].median())
leads_df['Lead Source'] = leads_df['Lead Source'].fillna('Unknown')
leads_df['Last Activity'] = leads_df['Last Activity'].fillna('Unknown')

# Überprüfen auf verbleibende NaN-Werte und ersetzen
leads_df.fillna(0, inplace=True)

# Unnötige Spalten entfernen
leads_df.drop(['Prospect ID', 'Lead Number', 'City', 'Lead Profile'], axis=1, inplace=True)

# Kategoriale Daten in Strings umwandeln, um Konsistenz sicherzustellen
for column in leads_df.select_dtypes(include=['object']).columns:
    leads_df[column] = leads_df[column].astype(str)

# Kategoriale Daten in numerische Werte umwandeln
label_encoder = LabelEncoder()
for column in leads_df.select_dtypes(include=['object']).columns:
    leads_df[column] = label_encoder.fit_transform(leads_df[column])

# Ziel- und Feature-Variablen definieren
y = leads_df['Converted']
x = leads_df.drop(['Converted'], axis=1)

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Daten skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modell initialisieren und trainieren
model = LogisticRegression(max_iter=5000, solver='saga', random_state=42)
model.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# Lead-Score in DataFrame einfügen
leads_df['Lead_Score'] = model.predict_proba(scaler.transform(x))[:, 1]

# High-Potential-Leads in CSV exportieren
high_potential = leads_df[leads_df['Lead_Score'] > 0.7]
high_potential.to_csv('high_potential_leads.csv', index=False)

# Neo4j-Verbindung aufbauen (Knowledge Graph)
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Knowledge Graph befüllen
for index, row in high_potential.iterrows():
    lead_node = Node("Lead", name=f"Lead_{index}", score=float(row['Lead_Score']))
    graph.merge(lead_node, "Lead", "name")

    country_node = Node("Country", name=str(row['Country']))
    graph.merge(country_node, "Country", "name")

    relationship = Relationship(lead_node, "LOCATED_IN", country_node)
    graph.merge(relationship)

# Modell evaluieren
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Genauigkeit: {accuracy:.2f}")
print("Konfusionsmatrix:\n", conf_matrix)
print("Bericht:\n", report)
print("High-Potential-Leads wurden in 'high_potential_leads.csv' exportiert.")

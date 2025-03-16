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

# Ursprüngliche Kategorie-Mappings speichern
category_mappings = {}

# Kategoriale Daten in Strings umwandeln und Mapping erstellen
for column in leads_df.select_dtypes(include=['object']).columns:
    leads_df[column] = leads_df[column].astype(str)
    unique_values = leads_df[column].unique()
    category_mappings[column] = {value: idx for idx, value in enumerate(unique_values)}
    leads_df[column] = leads_df[column].map(category_mappings[column])

# Mapping speichern
print("Kategorie-Mappings:")
for key, value in category_mappings.items():
    print(f"{key}: {value}")

# Kategorie-Mappings in CSV speichern
with open('category_mappings.csv', 'w', newline='') as csvfile:
    fieldnames = ['Feature', 'Category', 'Encoded Value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for column, mapping in category_mappings.items():
        for label, encoded_value in mapping.items():
            writer.writerow({'Feature': column, 'Category': label, 'Encoded Value': encoded_value})

print("Kategorie-Mappings wurden in 'category_mappings.csv' gespeichert.")

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

# Vorhersagen mit dem Testset
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# Lead-Score in DataFrame einfügen
leads_df['Lead_Score'] = model.predict_proba(scaler.transform(x))[:, 1]

# High-Potential-Leads in CSV exportieren
high_potential = leads_df[leads_df['Lead_Score'] > 0.7]
high_potential.to_csv('high_potential_leads.csv', index=False)

# Neo4j-Verbindung aufbauen (Knowledge Graph)
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
# Alle bestehenden Daten in Neo4j löschen
graph.run("MATCH (n) DETACH DELETE n")


# Knowledge Graph mit mehreren Kategorien befüllen
for index, row in high_potential.iterrows():
    lead_node = Node("Lead", name=f"Lead_{index}", score=float(row['Lead_Score']))
    graph.merge(lead_node, "Lead", "name")

    # Lead Source als Knoten
    source_name = [k for k, v in category_mappings['Lead Source'].items() if v == row['Lead Source']][0]
    source_node = Node("LeadSource", name=source_name)
    graph.merge(source_node, "LeadSource", "name")
    graph.merge(Relationship(lead_node, "SOURCE_FROM", source_node))

    # Last Activity als Knoten
    activity_name = [k for k, v in category_mappings['Last Activity'].items() if v == row['Last Activity']][0]
    activity_node = Node("LastActivity", name=activity_name)
    graph.merge(activity_node, "LastActivity", "name")
    graph.merge(Relationship(lead_node, "LAST_INTERACTION", activity_node))

    # Page Views als Engagement-Knoten
    engagement_node = Node("Engagement", page_views=int(row['Page Views Per Visit']))
    graph.merge(engagement_node, "Engagement", "page_views")
    graph.merge(Relationship(lead_node, "HAS_ENGAGEMENT", engagement_node))

    # Asymmetrique Activity Score
    score_node = Node("ActivityScore", score=float(row['Asymmetrique Activity Score']))
    graph.merge(score_node, "ActivityScore", "score")
    graph.merge(Relationship(lead_node, "HAS_SCORE", score_node))

# Modell evaluieren
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Genauigkeit: {accuracy:.2f}")
print("Konfusionsmatrix:\n", conf_matrix)
print("Bericht:\n", report)
print("High-Potential-Leads wurden in 'high_potential_leads.csv' exportiert.")

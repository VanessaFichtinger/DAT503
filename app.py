import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
leads_df.fillna({
    'Country': 'Unknown',
    'TotalVisits': leads_df['TotalVisits'].median(),
    'Page Views Per Visit': leads_df['Page Views Per Visit'].median(),
    'Lead Source': 'Unknown',
    'Last Activity': 'Unknown',
    'Lead Quality': 'Unknown',
    'Total Time Spent on Website': 0
}, inplace=True)

# Falls noch NaN-Werte vorhanden sind, diese mit 0 ersetzen
leads_df.fillna(0, inplace=True)

# Unnötige Spalten entfernen
leads_df.drop(['Prospect ID', 'Lead Number', 'City', 'Lead Profile'], axis=1, inplace=True)

# 'Yes' → 1, 'No' → 0
leads_df['Do Not Email'] = leads_df['Do Not Email'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
leads_df['Do Not Call'] = leads_df['Do Not Call'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

# Neue Features erstellen (Feature Engineering)
leads_df['Engagement_Score'] = leads_df['TotalVisits'] * leads_df['Total Time Spent on Website']
leads_df['Contactability_Score'] = (1 - leads_df['Do Not Email']) * (1 - leads_df['Do Not Call'])

# Ursprüngliche Kategorie-Mappings speichern
category_mappings = {}

# LabelEncoder initialisieren
le = LabelEncoder()

# Spalten identifizieren, die keine numerischen Werte enthalten
categorical_columns = leads_df.select_dtypes(include=['object']).columns.tolist()

# 'Do Not Email' und 'Do Not Call' entfernen, weil sie schon numerisch sind
categorical_columns = [col for col in categorical_columns if col not in ['Do Not Email', 'Do Not Call']]

# Label Encoding nur auf verbleibende kategoriale Spalten anwenden
for column in categorical_columns:
    # Sicherstellen, dass alles Strings sind
    leads_df[column] = leads_df[column].astype(str)
    
    # Fitting des LabelEncoders
    le.fit(leads_df[column])
    
    # Speichern der Mappings
    category_mappings[column] = {label: idx for idx, label in enumerate(le.classes_)}
    
    # Anwenden der Umwandlung auf die Spalte
    leads_df[column] = le.transform(leads_df[column])

# Überprüfe die gespeicherten Mappings
print("Category Mappings:\n", category_mappings)

# Zielvariable und Features definieren
y = leads_df['Converted']
X = leads_df.drop(['Converted'], axis=1)

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Daten skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Modell trainieren
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Lead-Score berechnen
leads_df['Lead_Score'] = model.predict_proba(scaler.transform(X))[:, 1]

# High-Potential-Leads filtern & speichern
high_potential = leads_df[leads_df['Lead_Score'] > 0.7]
high_potential.to_csv('high_potential_leads.csv', index=False)

# Feature Importance ausgeben
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Top Features:\n", feature_importances.head(10))

# Neo4j-Verbindung aufbauen (Knowledge Graph)
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
# Alle bestehenden Daten in Neo4j löschen
graph.run("MATCH (n) DETACH DELETE n")

# Knowledge Graph befüllen
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

    # Engagement Score als Knoten
    engagement_node = Node("Engagement", score=float(row['Engagement_Score']))
    graph.merge(engagement_node, "Engagement", "score")
    graph.merge(Relationship(lead_node, "HAS_ENGAGEMENT", engagement_node))

# Modell evaluieren
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Genauigkeit: {accuracy:.2f}")
print("Konfusionsmatrix:\n", conf_matrix)
print("Bericht:\n", report)
print("High-Potential-Leads wurden in 'high_potential_leads.csv' exportiert.")

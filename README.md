# DAT503
Prototyp für Vertriebsautomatisierung durch Potenzialberechnung für Einnahmen

Verwendeter Datensatz: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset/data 

Wir haben zwei unterschiedliche Modelle mit dem Datensatz getestet.

Für beide implementationen wurden der Datensatz in Form eines CSVs eingelesen, die Daten anschließend vorbereitet und ein Modell mithilfe der sklearn Bibilothek trainiert.
Ebenfalls wurden die Bibiliotheken pandas, numpy, csv und py2neo verwendet.

In dem Branch Main wurde Logistische Regression verwendet und anschließend anhand der Vorhersage die High-Potential-Leads in eine CSV-Datei exportiert.
Ebenfalls wird hier eine Verbindung mit einer Neo4j Datenbank erstellt um dort anschließend Knowledge Graphs erstellen zu können.
Am Ende des Programms wird auch noch eine kurze Modellbewertung durchgeführt.

In dem Branch random_forest wurde das Random Forest Modell verwendet und ebenfalls anschließend die High-Potential-Leads in eine CSV-Datei exportiert, 
die Daten an die Neo4j Datenbank gesendet und abschließend eine kurze Modellbewertung durchgeführt.

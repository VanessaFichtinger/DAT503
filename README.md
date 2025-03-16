# DAT503
Prototyp für Vertriebsautomatisierung durch Potenzialberechnung

Verwendeter Datensatz: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset/data

Wir haben zwei unterschiedliche Modelle mit dem Datensatz getestet.

Für beide implementationen wurden der Datensatz in Form eines CSVs eingelesen, die Daten anschließend vorbereitet, Mappings erstellt und als csv exportiert und ein Modell mithilfe der sklearn Bibilothek trainiert. Ebenfalls wurden die Bibiliotheken pandas, numpy, csv und py2neo verwendet.

In dem Branch Main wurde Logistische Regression verwendet und anschließend anhand der Vorhersage die High-Potential-Leads in eine CSV-Datei exportiert. Ebenfalls wird hier eine Verbindung mit einer Neo4j Datenbank erstellt um dort anschließend Knowledge Graphs erstellen zu können. Am Ende des Programms wird auch noch eine kurze Modellbewertung durchgeführt.

In dem Branch random_forest wurde das Random Forest Modell verwendet und ebenfalls anschließend die High-Potential-Leads in eine CSV-Datei exportiert, die Daten an die Neo4j Datenbank gesendet und abschließend eine kurze Modellbewertung durchgeführt.

Ergebnisse: Beide branches exportieren ein CSV mit den High-Potential-Leads, dies sind Leads mit einem Lead Score über 0,7. Diese csv kann mithilfe der exportierten Mappings, ebenfalls in CSV form gelesen und analysiert werden. Ebenfalls werden diese High-Potential-Leads an eine lokale Neo4j Datenbank gesendet, dort kann man anschließend die unterschiedlichen Attribute der Leads und deren Beziehungen übersichtlich in einem Knowledge Graph darstellen. (Hierfür wird eine lokale Datenbank mit der Url: 'bolt://localhost:7687' Benutzer: 'neo4j' und Passwort: 'password' benötigt ansonsten muss der Code angepasst werden).

Die Auswertung der jeweiligen Modellbewertungen am Ende der Programme zeigen das insgesamt das Random Forest Modell genauere Ergebnisse liefert.

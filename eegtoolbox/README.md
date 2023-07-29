# EEGToolbox
Projekt Toolbox, welche alle Netze beinhaltet. 

Aufgeteilt in Klassen zur Erleichterung verteilter Arbeit.

Pipeline-Schritte Projekt-intern nach Ausführungsreihenfolge in Ordnern organisiert.

In allen Ordnern jeweils zur Dokumentation und für Projekt-ToDo's READMEs enthalten.

Abschnitte für:
- Datenimport
- Preprocessing 
- Visualisierung der Daten
- Netzauswahl
- Visualisierung der Ergebnisse
- Dokumentation der Ergebnisse mit Zeitstempel

## Überblick
Mithilfe dieser Toolbox soll der Arbeitsablauf vereinfacht werden. 

Aus der main-Funktion heraus können verschiedene Netze aufgerufen werden. 

Parallel kann über GIT jeder an seinen Netzen weiterarbeiten und pushen, 
ohne Konflikte zu erzeugen. 

Zentrale Punkte wie Datenimporte, Visualisierung und Dokumentation der Ergebnisse werden so für alle Netze standardisiert verfügbar.

## Struktur des Projektes

1. Datenimport
2. Filtering, Aufteilung der Daten für Training und Testen, anpassbar an benötigte Netzdimensionen
3. Datenvisualisierung
4. Auswahl des zu trainierenden Netzes / der zu trainierenden Netze
5. Visualisierung der Ergebnisse
6. Dokumentation der Ergebnisse

### *Anmerkungen*

1. Zunächst Beschränkung auf BCI Competition 2a Datensatz. "BCICIV_2a_gdf" Ordner-Inhalt einzeln hinzufügen, ***nicht*** zu GIT adden.
2. Shuffle-Optionen und Auswahl der Menge an Daten, (DataAugmentation), (Cross Validation)
3. Orientierung an SigViewer
4. EEGNet, MLP, Transformer, (InceptionNet), (FCN)
5. Accuracy und Loss Kurven für Training und Validierung, Accuracy Scores auf Testdaten, Confusion Matrix
6. Speichern mit Zeitstempel, (Netzkennwerte) *z.B.:  2022_07_02__15h41m31s__EEGNet_FullData_AccCurve.png*
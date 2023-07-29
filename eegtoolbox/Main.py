########################################################################
########################################################################
#   MAIN-FUNKTION DES PROJEKTS
#   Von hier aus werden alle Schritte der Pipeline aufgerufen
#   Datensatz sowie zu trainierendes Netz hier auswählbar
########################################################################
########################################################################

from _01_DataImport.DataImport import DataImport
from _02_Preprocessing.Preprocessing import Preprocessing
from _03_DataVisualisation.DataVisualisation import DataVisualisation
from _04_Networks.NetworkSelection import NetworkSelection
from _05_Evaluation.Evaluation import Evaluation

########################################################################
#   DATENSATZ UND NEURONALES NETZWERK AUSWAHL
#   Neue Datensätze oder Netze zu Liste hinzufügen nach Schema
#   Danach über Index aufrufbar
#   Bitte nur getestete Datensätze/Netze in main-branch committen
########################################################################

dataset_names = ['BCICIV_2a_Multiple',  # 0
                 'BCICIV_2a_Single'     # 1
                 ]
network_names = ['EEGNet',              # 0
                 'MLPKeras',            # 1
                 'MLPClassifier',       # 2
                 'Transformer',         # 3 ### START AT OWN RISK, CAN CRASH CPU
                 'FCN'                  # 4
                 #'InceptionNet',       # 5
                 ]

# Auswahl Dataset + Netz - hier die Zahlen ändern für andere Netze
selected_dataset = dataset_names[0]     # z.B. dataset_names[0] für BCICIV_2a_Multiple
selected_network = network_names[1]     # z.B. network_names[0] für EEGNet

print("Selected_Dataset: ", selected_dataset)
print("Selected_Network: ", selected_network)

########################################################################
#   PIPELINE
#   Bitte nichts ohne Absprache verändern
#   Konvention: immer nur 1 Aufruf pro Pipeline-Schritt
########################################################################

# Datenimport
dataimport = DataImport()
raws = dataimport.loadGDFFiles(selected_dataset)

# Preprocessing
preprocessing = Preprocessing()
X, y, splitted_values = preprocessing.preprocessing(raws, selected_network)

# Visualisierung der Daten
datavisualisation = DataVisualisation()
datavisualisation.visualise(X, y, splitted_values, selected_dataset)

# Auswahl des Netzwerks
networkselection = NetworkSelection()
pred, y_test, history, num_epochs, acc = networkselection.selectNetwork(selected_network, splitted_values)

# Visualisierung der Ergebnisse
evaluation = Evaluation()
evaluation.plot(selected_network, selected_dataset, pred, y_test, history, num_epochs, acc)


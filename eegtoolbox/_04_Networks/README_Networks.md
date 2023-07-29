# Network Selection

Auswahl des zu trainierenden Netzes. Zum Hinzufügen eines neuen Netzes muss Netz hier ausgewählt werden.

*Eintrag des Netzes in main-Funktion (siehe Kommentare in File) 
sowie in Evaluation.py (siehe README_Evaluation.md) notwendig*

## Template zum Netzwerk hinzufügen
*Für neue Netze elif-Klausel in NetworkSelection.py hinzufügen.*

*Sollte auch in NetworkSelection.py als auskommentiert enthalten sein.*
``` 
elif network_name == 'MyNewNetwork':
            network = NameOfPythonFile()
            pred, y_test, history, loss_curve, num_epochs = network.runFunctionForMyModel(splitted_values)
            return pred, y_test, history, loss_curve, num_epochs
```

### *Anmerkungen*
- 'MyNewNetwork'
  - Name des Netzwerks, wie er in main-Funktion bei network_names hinzugefügt wurde
- NameOfPythonFile() 
  - Name des Python File in _04_Networks/MyNetworkFolder Ordner mit "()" statt ".py" am Ende
- runFunctionForMyModel(splitted_values)
  - Funktion in eigenem Modul, die das Modell enthält

## Code-Beispiel eigene Klasse und Funktion
*Aus MLP Implementierung, für eigenes Netz anzupassen*

```
from _04_Networks.MLP import MLP


class MLP_without_FBCSPToolbox:

    def runMLPClassifier(self, splitted_values):

        # Training MLP Classifier
        print("------MLP Classifier Training-------")
```
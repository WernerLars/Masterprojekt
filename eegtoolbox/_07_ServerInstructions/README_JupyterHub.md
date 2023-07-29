# JupyterHub

Die folgenden Schritte sind notwendig, um die EEGToolbox in JupyterHub zu verwenden.

1. Ordnerstruktur inklusive Datensätzen hochladen
2. Falls noch nicht vorhanden, aus dem Terminal heraus mne-package installieren mit ```pip install mne```
3. Ausführen aus dem Terminal. Dafür gegebenfalls mit ```cd```-Befehl ins erstellte Verzeichnis wechseln und 
mit ```python Main.py``` ausführen.
4. Falls JupyterHub Probleme mit graphviz/pydot hat,
```keras.utils.plot_model(model=model, to_file='_04_Networks/Transformer/Models/transformer22.png', show_shapes=True)``` 
auskommentieren in folgenden Files:
   1. _04_Networks/Transformer/Transformer_22Channels.py
   2. _04_Networks/FCN/FCN_22Channels.py
5. Falls "ImportError: cannot import name 'to_categorical' from 'keras.utils'" auftritt, in _04_Networks/MLP/MLP.py 
   1. ```from keras.utils import to_categorical``` ändern zu ```from tensorflow.keras.utils import to_categorical```



Alternative zum Uni-Server (funktioniert mit EEGNet und MLP, für FCN/Transformer nicht genug Power) 
ist [Gradient](https://www.gradient.run):
1. Account erstellen
2. Neues Projekt anlegen und neues Notebook darin erstellen und öffnen
3. Auf ```Start Machine``` klicken um Server zu starten
4. Links unten auf kleines JupyterHub Symbol ```Open in JupyterHub``` klicken
5. Jetzt wie auf Uniserver Dateien hochladen (geht hier auch mit Datensätzen ohne Umweg)
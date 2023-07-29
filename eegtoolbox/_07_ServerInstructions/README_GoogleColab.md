# Google-Colab

Die folgenden 3 Schritte sind notwendig, um die EEGToolbox in JupyterHub zu verwenden.

1. Ordnerstruktur inklusive Datensätzen hochladen in GoogleDrive
2. Alle ```keras.util (...)``` imports zu ```tensorflow.keras.utils (...)``` umschreiben, zb in
   1. _04_Networks/MLP/MLP_without_FBCSPToolbox.py 
      1. ```from keras.utils import to_categorical``` ändern in 
      2. ```from tensorflow.keras.utils import to_categorical```
3. EEGToolbox.ipynb ausführen. Gegenenfalls Upload-Ort an eigenen DriveOrdner anpassen bei 
   1. ```%cd /content/drive/MyDrive/ColabNotebooks/EEGToolbox/```
# Evaluation

Plotten der Ergebnisse der jeweiligen Netze. 

Für neue Netze muss eigenes Netz zu Modell Liste in Evaluation.py hinzugefügt werden:

```
sklearn_models  = ['MLPClassifier']
keras_models    = ['EEGNet',
                   'FCN',
                   'InceptionNet',
                   'MLPKeras',
                   'Transformer']
```

- Accuracy Curves
- Loss Curves
- Confusion Matrices

- Integrierte Modellnamen und Zeitstempel für konsekutive eindeutige Benennung
- Unterscheidung nach Sklearn oder Keras Modellen bei Accuracy und Loss Curves

### To Do
- Plot Accuracy Curves for sklearn Models implementieren + Validity Curves hinzufügen
- Ausgabe von Netzparametern in Textform implementieren
- Hinzufügen des eigenen Netzes vereinfachen / automatisieren aus main heraus wäre evtl besser

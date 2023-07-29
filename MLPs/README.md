# Welcome to the MLP Project 
- This Project trains and tests MLP Networks on EEG Competition 2a Dataset
- Generally we have two kinds if networks, namely MLPClassifier from scitkit learn and Keras Models
- There are two Main Python Files to use to train and test MLP Networks
  - EEG_with_MLP uses FBCSPToolbox to extract Features
  - MLP_without_FBCSPToolbox directly trains on imported data
- MNE is used to import the GDF Files of the Dataset
- Graphs for Evaluation are saved to a figures folder

# File Descriptions
- FBCSPToolbox Files
  - only used to Filter and Extract Data in EEG_with_MLP
  - only binary classification after applying this Toolbox

- DataAugmentation
  - can be used to get more data from a class
  - used in EEG_with_MLP, because FBCSPToolbox only allows binary classification
  - we have 4 classes, so that would mean to have 1/4 True Positive and 3/4 True negative

- GridSearch
  - tried to apply GridSearch for MLPClassifier and Keras Models
  - sklearn has GridSearchCV for MLPClassifier
  - Keras Models need KerasClassifier, but it doesnt work (some DeepCopy Problem of the model)
  
- PlotGraphs
  - generates Graphs for Evaluation in the figures folder
  - in can generate Confusion Matrix for both network types
  - for MLPClassifier, only loss curve is available
  - for keras model, we can print out accuracy and loss curves

- Main Files:
  - EEG_with_MLP
    - only binary classification because of FBCSPToolbox
    - Loop for each class, which means we create 4 MLPClassifier and 4 Keras Models
  - MLP_without_FBCSP
    - uses directly data from epochs
    - one File consists of over 500 Samples with each 22 channels and 2376 Points of the curve
    - networks need (samples, curve points) so we have to create new labels
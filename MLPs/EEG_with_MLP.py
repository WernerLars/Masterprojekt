import mne
import numpy as np
from tensorflow import keras
from FBCSPToolbox.FBCSP import FBCSP
from FBCSPToolbox.Filterbank import Filterbank
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import MLP
import DataAugmentation
import GridSearch
import PlotGraphs

# Loading Data with mne
folder_path_string = "BCICIV_2a_gdf/"

# First one for fewer data import
filenames = ["A01T.gdf", 'A02T.gdf']
#filenames = ["A01T.gdf", 'A02T.gdf', 'A03T.gdf', 'A05T.gdf', 'A06T.gdf', 'A07T.gdf', 'A08T.gdf', 'A09T.gdf']

# Load in GDF Data Files
raws = []
for filename in filenames:
    path = folder_path_string + filename
    raws.append(mne.io.read_raw_gdf(path))

# Print some Information about the imported data
print(raws[0])
data_info = raws[0].info
print(data_info)
# data.plot(duration=5, n_channels=30)

# Preprocessing
counter = 1
fs = 0
X = []
y = []
window_details = {'tmin': 0.5, 'tmax': 2.5}
for data in raws:
    fs = data.info.get('sfreq')
    events, event_ids = mne.events_from_annotations(data)
    stims = [value for key, value in event_ids.items() if key in ('769', '770', '771', '772')]
    epochs = mne.Epochs(data, events, event_id=stims, tmin=-4.5, tmax=5.0, event_repeated='drop',
                        baseline=None, preload=True, proj=False, reject_by_annotation=False)
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
    epochs = epochs.drop_channels(channels_to_remove)

    # Concatenate epoch data
    if counter == 1:
        X = epochs.get_data() * 1000
        y = epochs.events[:, -1] - min(epochs.events[:, -1])

    else:
        X = np.concatenate([X, epochs.get_data() * 1000])
        y = np.concatenate([y, epochs.events[:, -1] - min(epochs.events[:, -1])])

    counter += 1

# Shuffle all Data
X,y = shuffle(X,y,random_state=0)
eeg_data = {'x_data': X,
            'y_labels': y,
            'fs': fs}

# Filter Data with Filterbank
filtered_data = Filterbank(eeg_data, window_details, fs)

# Split Data in Training and Test Set
split_ratio = 0.75
train_idx = round(len(y) * split_ratio)

X_train = np.copy(filtered_data[:, :train_idx, :, :])
X_test = np.copy(filtered_data[:, train_idx:, :, :])
y_train = np.copy(y[:train_idx])
y_test = np.copy(y[train_idx:])

print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)

y_classes_unique = np.unique(y_train)
n_classes = len(np.unique(y_train))
print("Number of Unique Classes: ", n_classes)
print("Unique Classes: ", y_classes_unique)

print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# Feature Extraction with FBCSP
fbcsp = FBCSP(2)
fbcsp.fit(X_train, y_train)


# Training MLP Classifier
print("------MLP Classifier Training-------")

for j in range(n_classes):
    cls_of_interest = y_classes_unique[j]
    select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

    y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
    y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

    x_features_train = fbcsp.transform(X_train, class_idx=cls_of_interest)
    x_features_test = fbcsp.transform(X_test, class_idx=cls_of_interest)

    print("Training Data before Augmentation: ")
    count_tp_tn = DataAugmentation.count_and_print_tp_tn(y_train_cls)

    # Augmentation
    x_features_train, y_train_cls = DataAugmentation.augmentation(count_tp_tn, x_features_train, y_train_cls)

    print("Training Data after Augmentation: ")
    DataAugmentation.count_and_print_tp_tn(y_train_cls)

    # Creating MLP Classifier with sklearn
    mlp_classifier = MLPClassifier(hidden_layer_sizes=[32, 16, 4], max_iter=1500, learning_rate='constant',
                                   solver='sgd', alpha=0.00001)

    # Grid Search for MLP Classifier
    #GridSearch.GridSearchMLPClassifier(mlp_classifier, x_features_train, y_train_cls)

    mlp_classifier.fit(x_features_train, np.asarray(y_train_cls, dtype=float))
    # For evaluating sklearn MLP Classifier
    pred = mlp_classifier.predict(x_features_test)
    cm = confusion_matrix(pred, y_test_cls)
    print("Prediction for Class :", cls_of_interest)
    count_tp_tn = DataAugmentation.count_and_print_tp_tn(y_test_cls)
    print(cm)
    print(mlp_classifier.score(x_features_test, y_test_cls))

    PlotGraphs.plotConfusionMatrix('mlpclassifier',cls_of_interest,pred,y_test_cls)
    PlotGraphs.plotGraphsSklearn('mlpclassifier',cls_of_interest, mlp_classifier.loss_curve_)

# Training Keras MLP
print("------MLP Keras Training-------")

for j in range(n_classes):
    cls_of_interest = y_classes_unique[j]
    select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

    y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
    y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

    x_features_train = fbcsp.transform(X_train, class_idx=cls_of_interest)
    x_features_test = fbcsp.transform(X_test, class_idx=cls_of_interest)

    print("Training Data before Augmentation: ")
    count_tp_tn = DataAugmentation.count_and_print_tp_tn(y_train_cls)

    # Augmentation
    x_features_train, y_train_cls = DataAugmentation.augmentation(count_tp_tn, x_features_train, y_train_cls)

    print("Training Data after Augmentation: ")
    DataAugmentation.count_and_print_tp_tn(y_train_cls)

    # Create and Evaluate MLP over Keras
    mlp_keras = MLP.createMLP()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    mlp_keras.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Grid Search with KerasClassifier (does not work)
    #GridSearch.GridSearchKerasClassifier(mlp_keras, x_features_train, y_train_cls)

    history = mlp_keras.fit(x_features_train, np.asarray(y_train_cls, dtype=float), epochs=30)

    # Evaluating Keras Model
    print("Prediction for Class :", cls_of_interest)
    count_tp_tn = DataAugmentation.count_and_print_tp_tn(y_test_cls)

    pred = mlp_keras.predict(x_features_test).reshape(-1)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    cr = classification_report(pred, y_test_cls, target_names=['0', '1'])
    print(cr)

    PlotGraphs.plotConfusionMatrix('keras_model',cls_of_interest,pred,y_test_cls)
    PlotGraphs.plotGraphsKeras('keras_model',cls_of_interest,history,0,30,0,1)

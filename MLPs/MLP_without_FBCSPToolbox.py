import mne
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
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
window_size = {'tmin': -4.5, 'tmax': 5.0}
for data in raws:
    fs = data.info.get('sfreq')
    events, event_ids = mne.events_from_annotations(data)
    stims = [value for key, value in event_ids.items() if key in ('769', '770', '771', '772')]
    epochs = mne.Epochs(data, events, event_id=stims, tmin=window_size['tmin'], tmax=window_size['tmax'], event_repeated='drop',
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

print("X and y shapes before Transformation: ")
print("X.shape: ", X.shape)
print("y.shape: ", y.shape)

x_transform = []
y_transform = []
counter = 1
for i in range(X.shape[0]):
    for j in range(round(X.shape[1]/11)):
        if counter == 1:
            x_transform = [X[i, j, :]]
            y_transform = [y[i]]
        else:
            x_transform = np.concatenate([x_transform, [X[i, j, :]]])
            y_transform = np.concatenate([y_transform, [y[i]]])
    counter += 1

# Shuffle all Data
x_transform, y_transform = shuffle(x_transform, y_transform, random_state=0)

print("X and y shapes after Transformation: ")
print("X.shape: ", x_transform.shape)
print("y.shape: ", y_transform.shape)

# Split Data in Training and Test Set
split_ratio = 0.75
train_idx = round(len(y_transform) * split_ratio)

x_train = np.copy(x_transform[:train_idx, :])
x_test = np.copy(x_transform[train_idx:, :])
y_train = np.copy(y_transform[:train_idx])
y_test = np.copy(y_transform[train_idx:])

print("X_train shape :", x_train.shape)
print("y_train shape :", y_train.shape)
print("X_test shape :", x_test.shape)
print("y_test shape :", y_test.shape)

# Training MLP Classifier
print("------MLP Classifier Training-------")

# Creating MLP Classifier with sklearn
mlp_classifier = MLPClassifier(hidden_layer_sizes=[1024, 256, 64, 8], max_iter=1500, learning_rate='constant',
                                solver='adam', alpha=0.00001)

# Grid Search for MLP Classifier
#GridSearch.GridSearchMLPClassifier(mlp_classifier, x_train, y_train)

mlp_classifier.fit(x_train, np.asarray(y_train, dtype=float))
# For evaluating sklearn MLP Classifier
pred = mlp_classifier.predict(x_test)
cm = confusion_matrix(pred, y_test)

print(cm)
print(mlp_classifier.score(x_test, y_test))

PlotGraphs.plotConfusionMatrix('mlpclassifier','all',pred,y_test)
PlotGraphs.plotGraphsSklearn('mlpclassifier','all', mlp_classifier.loss_curve_)

# Training Keras MLP
print("------MLP Keras Training-------")

y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)


# Create and Evaluate MLP over Keras
mlp_keras = MLP.createMLP2376()
opt = keras.optimizers.Adam(learning_rate=0.01)
mlp_keras.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Grid Search with KerasClassifier (does not work)
#GridSearch.GridSearchKerasClassifier(mlp_keras, x_features_train, y_train_cls)

history = mlp_keras.fit(x_train, np.asarray(y_train, dtype=float), epochs=30)

# Evaluating Keras Model

pred = mlp_keras.predict(x_test)

PlotGraphs.plotConfusionMatrix('keras_model','all',pred.argmax(axis=1),y_test.argmax(axis=1))
PlotGraphs.plotGraphsKeras('keras_model','all',history,0,30,0,1)
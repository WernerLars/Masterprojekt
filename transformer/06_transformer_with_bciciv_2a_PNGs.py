'''

pip install mne
pip install pydot

watch -n 0.1 nvidia-smi


cd work && python 06_transformer_with_bciciv_2a_PNGs.py


'''


import tensorflow as tf

from tensorflow import keras

## Data import
import numpy as np
import mne

# EEGNet-specific imports
from tensorflow.keras import utils as np_utils
from keras import backend as K

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import os
import time
'''
from numba import cuda

### Clean up GPU usage
for x in range(3):
    cuda.select_device(x)
    cuda.close()
'''    

'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  #disable for tensorFlow V2
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.experimental.set_memory_growth(physical_devices[2], True)
tf.config.experimental.set_memory_growth(physical_devices[3], True)
'''


CUDA_VISIBLE_DEVICES="0,1,2,3"
allow_growth=True
TF_GPU_ALLOCATOR='cuda_malloc_async'

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


folder_path_string = "BCICIV_2a_gdf/"

file_path_string = folder_path_string + "A01T.gdf"
evaluation_path_string = folder_path_string + "A01T.evt"

raws = []

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

##################### Process, filter and epoch the data ######################
raw_fname = file_path_string
event_fname = evaluation_path_string
tmin, tmax = -0., 1

# Setup for reading the raw data
raw = mne.io.read_raw_gdf(raw_fname)
raw.load_data()

#original filter
raw.filter(2, None, method='iir')  # replace baselining with high-pass

events, event_ids = mne.events_from_annotations(raw)
event_id = events[1]
print("event_id ",event_id)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

# Read epochs
stims =[value for key, value in event_ids.items() if key in ('769','770','771','772')]
epochs = mne.Epochs(raw, events, stims, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
epochs = epochs.drop_channels(channels_to_remove)
labels = epochs.events[:, -1]
print(epochs)

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels
y[y == 7] = 0
y[y == 8] = 1
y[y == 9] = 2
y[y == 10] = 3

kernels, chans, samples = 1, X.shape[1], X.shape[2]

# take 50/25/25 percent of the data to train/validate/test

train_idx = round(X.shape[0] * 0.5)
test_idx = round(X.shape[0] * 0.75)

X_train      = X[0:train_idx,]
Y_train      = y[0:train_idx]
X_validate   = X[train_idx:test_idx,]
Y_validate   = y[train_idx:test_idx]
X_test       = X[test_idx:,]
Y_test       = y[test_idx:]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
y_unique = np.unique(y)
num_classes = len(y_unique)
Y_train      = np_utils.to_categorical(Y_train-1, num_classes=num_classes)
Y_validate   = np_utils.to_categorical(Y_validate-1, num_classes=num_classes)
Y_test       = np_utils.to_categorical(Y_test-1, num_classes=num_classes)

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print("X.Shape", X.shape)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print("y.Shape", y.shape)
print('y_train shape:', Y_train.shape)
print('y_test shape:', Y_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print('num_classes:', num_classes)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x) #channels_first
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

## Train and Evaluate

input_shape = X_train.shape[1:]

model = build_model(
    input_shape,
    head_size=64,#256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

keras.utils.plot_model(model, show_shapes=True)


# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-2),
              metrics = ["categorical_accuracy"])

#model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_transformer_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True
    )
]


history = model.fit(
    X_train,
    Y_train,
    validation_split=0.2,
    epochs=2,
    batch_size=5,#64, 5 Maximum mit einer GPU auf JupyterLab
    callbacks=callbacks,
)

### Test model on data

testmodel = keras.models.load_model("best_transformer_model.h5")
test_loss, test_acc = testmodel.evaluate(X_test, Y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)



timestr = time.strftime("%Y_%m_%d-%H_%M_%S")


### Categorical accuracy

plt.figure()
plt.plot(history.history["categorical_accuracy"])
plt.plot(history.history["val_categorical_accuracy"])
plt.title("model test accuracy: " + str(test_acc) )
plt.ylabel('categorical_accuracy', fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig('transformer_accuracy_' + timestr + '.png')

## Categorical loss

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model test loss: ' + str(test_loss) )
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('transformer_loss_' + timestr + '.png')


### Confusion matrix

pred = (testmodel.predict(X_test)).argmax(axis=1)
Y_test_cls = Y_test.argmax(axis=1)
cm = confusion_matrix(pred, Y_test_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusionmatrix' + timestr + '.png')
########################################################################
########################################################################
#   FCN - 22 Channel Version
#   Aus Timeseries Beispiel abgeleitet
#   Optimiert für Ford-Datensatz auf >95% accuracy
#   Evtl ja für BCICIV2a optimierbar
########################################################################
########################################################################
from keras.utils import np_utils
from tensorflow import keras
import numpy as np

from _04_Networks.FCN import FCN_model

class FCN_22Channels:

    def runFCN_22Channels(self, splitted_values):

        X_train = splitted_values['X_train']
        y_train = splitted_values['y_train']
        X_validate = splitted_values['X_validate']
        y_validate = splitted_values['y_validate']
        X_test = splitted_values['X_test']
        y_test = splitted_values['y_test']

        print("Shape: ", X_train.shape[1:])

        chans       = 22
        samples     = 251
        kernels = 1

        # convert labels to one-hot encodings.
        y_unique = np.unique(y_train)
        num_classes = len(y_unique)
        y_train = np_utils.to_categorical(y_train - 1, num_classes=num_classes)
        y_validate = np_utils.to_categorical(y_validate - 1, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test - 1, num_classes=num_classes)

        # convert data to NHWC (trials, channels, samples, kernels) format. Data
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
        ##############################################################################
        # HYPERPARAMETER
        ##############################################################################

        # Model
        num_classes = 4
        filters = 64
        kernel_size = 3

        # Fitting
        num_epochs = 10
        batch_size = 32
        #validation_split_percentage = 0.2

        ##############################################################################
        ##############################################################################

        model = FCN_model.build_fcn_model(input_shape=X_train.shape[1:], num_classes=num_classes,
                                          filters = filters, kernel_size = kernel_size)

        # keras.utils.plot_model(model=model, to_file='_04_Networks/FCN/Models/fcn22.png',
        #                        show_shapes=True)


        checkpointer = keras.callbacks.ModelCheckpoint(filepath='_04_Networks/FCN/Models/best_fcn_model.h5',
                                                       verbose=1, save_best_only=True)

        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics = ['accuracy', 'TruePositives', 'TrueNegatives'])

        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=checkpointer,
            validation_data=(X_validate, y_validate),
            #validation_split=validation_split_percentage,
            verbose=2, #1
        )

        # load optimal weights
        model.load_weights('_04_Networks/FCN/Models/best_fcn_model.h5')

        preds = (model.predict(X_test)).argmax(axis=1)
        y_test_cls = y_test.argmax(axis=1)

        acc         = np.mean(preds == y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))

        return preds, y_test_cls, history, num_epochs, acc

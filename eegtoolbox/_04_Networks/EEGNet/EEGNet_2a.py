########################################################################
########################################################################
#   EEGNET INTEGRATION VORLAGE
#   Abgeleitet von main-branch (Stand 02.07.2022)
#   Gesäubert um alle nicht benutzten Imports / Funktionen
#   Als Vorlage gedacht für eigene EEGNet Varianten
#   EEGNet-Model siehe EEGNet_model.py
########################################################################
########################################################################

import numpy as np
import tensorflow
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from _04_Networks.EEGNet import EEGNet_model

class EEGNet_2a:

    def runEEGNet(self, splitted_values):
        ##############################################################################
        # HYPERPARAMETER
        ##############################################################################

        # Model
        nb_classes  = 4
        chans       = 22
        samples     = 251
        dropoutRate = 0.5
        kernLength  = 128
        F1          = 16
        D           = 2
        F2          = 16
        dropoutType = 'Dropout'
        kernels = 1

        # Fitting
        batch_size  = 16
        num_epochs  = 100

        ##############################################################################
        ##############################################################################

        X_train = splitted_values['X_train']
        y_train = splitted_values['y_train']
        X_validate = splitted_values['X_validate']
        y_validate = splitted_values['y_validate']
        X_test = splitted_values['X_test']
        y_test = splitted_values['y_test']

        #set seed
        tensorflow.random.set_seed(1234)

        y_train      = np_utils.to_categorical(y_train-1, num_classes=nb_classes)
        y_validate   = np_utils.to_categorical(y_validate-1, num_classes=nb_classes)
        y_test       = np_utils.to_categorical(y_test-1, num_classes=nb_classes)

        # convert data to NHWC (trials, channels, samples, kernels) format. Data
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

        # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
        # model configurations may do better, but this is a good starting point)
        model = EEGNet_model.createEEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples,
                                          dropoutRate=dropoutRate, kernLength=kernLength,
                                          F1=F1, D=D, F2=F2, dropoutType=dropoutType)

        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics = ['accuracy', 'TruePositives', 'TrueNegatives'])

        # count number of parameters in the model
        numParams    = model.count_params()

        # set a valid path for your system to record model checkpoints
        checkpointer = ModelCheckpoint(filepath='_04_Networks/EEGNet/Models/checkpoint.h5', verbose=1,
                                       save_best_only=True)

        fittedModel = model.fit(x = X_train,
                                y = y_train,
                                batch_size=batch_size,
                                epochs = num_epochs,
                                verbose = 2,
                                validation_data=(X_validate, y_validate),
                                callbacks=[checkpointer])

        # load optimal weights
        model.load_weights('_04_Networks/EEGNet/Models/checkpoint.h5')

        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)
        acc         = np.mean(preds == y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % acc)

        #pred_cls = (model.predict(X_test)).argmax(axis=1)
        y_test_cls = y_test.argmax(axis=1)

        return preds, y_test_cls, fittedModel, num_epochs, acc

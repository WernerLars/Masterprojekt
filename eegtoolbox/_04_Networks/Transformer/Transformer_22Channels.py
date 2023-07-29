########################################################################
########################################################################
#   TRANSFORMER NET - 22 Channel Version
#   Lief bisher nur einigermaßen auf dem Uni-Jupyterhub
#   Auf privaten Rechnern besser vorher alles speichern
########################################################################
########################################################################
import numpy as np
from keras.utils import np_utils
from tensorflow import keras

from _04_Networks.Transformer import Transformer_model


class Transformer_22Channels:

    def runTransformer_22Channels(self, splitted_values):

        X_train = splitted_values['X_train']
        y_train = splitted_values['y_train']
        X_validate = splitted_values['X_validate']
        y_validate = splitted_values['y_validate']
        X_test = splitted_values['X_test']
        y_test = splitted_values['y_test']

        chans = 22
        samples = 251
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
        head_size = 64  # 256,
        num_heads = 4
        ff_dim = 4
        num_transformer_blocks = 2
        mlp_units = [128]
        mlp_dropout = 0.4
        dropout = 0.25
        num_classes = 4
        learning_rate = 1e-2

        # Fitting
        #validation_split_percentage = 0.2
        num_epochs = 1
        batch_size = 1  # 64, 5 Maximum mit einer GPU auf JupyterLab

        ##############################################################################
        ##############################################################################

        ## Train and Evaluate

        input_shape = X_train.shape[1:]

        model = Transformer_model.build_transformer_model(
            input_shape,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            mlp_dropout=mlp_dropout,
            dropout=dropout,
            num_classes=num_classes
        )

        # keras.utils.plot_model(model=model, to_file='_04_Networks/Transformer/Models/transformer22.png',
        #                        show_shapes=True)

        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics = ["accuracy"]) #"categorical_accuracy"

        model.summary()

        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath='_04_Networks/Transformer/Models/best_transformer_model.h5',
                                            save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_validate, y_validate),
            #validation_split=validation_split_percentage,
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        ### Test model on data

        #testmodel = keras.models.load_model('_04_Networks/Transformer/Models/best_transformer_model.h5')
        model.load_weights('_04_Networks/Transformer/Models/best_transformer_model.h5')
        test_loss, test_acc = model.evaluate(X_test, y_test)

        print("Test accuracy", test_acc)
        print("Test loss", test_loss)

        preds = (model.predict(X_test)).argmax(axis=1)
        y_test_cls = y_test.argmax(axis=1)

        acc = np.mean(preds == y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))

        return preds, y_test_cls, history, num_epochs, acc
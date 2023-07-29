import numpy as np
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from keras.utils import to_categorical
from _04_Networks.MLP import MLPModel


# History Class for MLPClassifier (Sklearn does not have one)

class History:
    history = {}

    def setHistory(self, loss, train_acc, val_loss, val_acc):
        self.history = {
            "loss": loss,
            "accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }


class MLP:

    def runMLPClassifier(self, splitted_values):
        # Training MLP Classifier
        print("------MLP Classifier Training-------")

        # Creating MLP Classifier with sklearn
        mlp_classifier = MLPClassifier(hidden_layer_sizes=[128, 64, 16],
                                       max_iter=1500, learning_rate='constant',
                                       solver='adam', alpha=0.00001)

        x_train = splitted_values['X_train']
        y_train = splitted_values['y_train']
        x_validate = splitted_values['X_validate']
        y_validate = splitted_values['y_validate']
        x_test = splitted_values['X_test']
        y_test = splitted_values['y_test']

        classes = np.unique(y_train)
        y_train = to_categorical(y_train, num_classes=4)
        y_validate = to_categorical(y_validate, num_classes=4)
        y_test = to_categorical(y_test, num_classes=4)

        train_acc = []
        val_acc = []
        val_loss = []
        num_epochs = 100
        epoch = 0
        while epoch < num_epochs:
            mlp_classifier.partial_fit(x_train, np.asarray(y_train, dtype=float), classes=classes)
            train_score = mlp_classifier.score(x_train, y_train)
            train_acc.append(train_score)
            test_score = mlp_classifier.score(x_validate, y_validate)
            val_acc.append(test_score)
            print("Epoch: ", epoch, "; Train Score: ", train_score, "; Test Score: ", test_score)

            val_pred = mlp_classifier.predict_proba(x_validate)
            loss = log_loss(y_validate, val_pred)
            val_loss.append(loss)
            epoch += 1

        history = History()
        history.setHistory(mlp_classifier.loss_curve_, train_acc, val_loss, val_acc)

        # For evaluating sklearn MLP Classifier
        pred = mlp_classifier.predict(x_test)
        acc = np.mean(pred.argmax(axis=-1) == y_test.argmax(axis=-1))
        print("Test accuracy: %f " % (acc))

        return pred.argmax(axis=1), y_test.argmax(axis=1), history, num_epochs, acc

    def runKerasModel(self, splitted_values):
        # Training Keras MLP
        print("------MLP Keras Training-------")

        x_train = splitted_values['X_train']
        y_train = splitted_values['y_train']
        x_validate = splitted_values['X_validate']
        y_validate = splitted_values['y_validate']
        x_test = splitted_values['X_test']
        y_test = splitted_values['y_test']

        y_train = to_categorical(y_train, num_classes=4)
        y_validate = to_categorical(y_validate, num_classes=4)
        y_test = to_categorical(y_test, num_classes=4)

        input_size = x_train.shape[1]
        print("x_train input size: ", input_size)

        # Create and Evaluate MLP over Keras
        mlp_keras = MLPModel.createMLP(input_size)
        opt = keras.optimizers.Adam(learning_rate=0.01)
        mlp_keras.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        num_epochs = 100
        history = mlp_keras.fit(x_train, np.asarray(y_train, dtype=float), epochs=num_epochs,
                                validation_data=(x_validate, y_validate))

        # Evaluating Keras Model
        pred = mlp_keras.predict(x_test)
        acc = np.mean(pred.argmax(axis=-1) == y_test.argmax(axis=-1))
        print("Test accuracy: %f " % (acc))

        return pred.argmax(axis=1), y_test.argmax(axis=1), history, num_epochs, acc

import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def GridSearchMLPClassifier(mlp_classifier, x_features_train, y_train_cls):
    # Parameter Space
    parameter_space = {
        'hidden_layer_sizes': [(32,16,8), (32,16,8,4,2), (32,8,4), (32,16,4)],
        'max_iter': [100,5000],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.00001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    # Grid Search
    gs = GridSearchCV(mlp_classifier, parameter_space)
    gs.fit(x_features_train,  np.asarray(y_train_cls, dtype=float))
    print("Best Parameters by Grid Search for this Class: ")
    print(gs.best_params_)

def GridSearchKerasClassifier(mlp_keras, x_features_train, y_train_cls):
    # Tried to apply GridSearchCV also for keras model, does not work yet

    # Parameter Space
    parameter_space = {'batch_size': [100, 20, 50, 25, 32],
              'nb_epoch': [200, 100, 300, 400],
              'unit': [5, 6, 10, 11, 12, 15],
            }

    # Grid Search
    mlp_keras_classifier = KerasClassifier(build_fn=mlp_keras)
    gs = GridSearchCV(mlp_keras_classifier, parameter_space)
    gs.fit(x_features_train,  np.asarray(y_train_cls, dtype=float))
    print("Best Parameters by Grid Search for this Class: ")
    print(gs.best_params_)


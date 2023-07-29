import mne
import numpy as np
from sklearn.utils import shuffle

from _02_Preprocessing.Transformation import Transformation


class Preprocessing:

    def preprocessing(self, raws, selected_network):

        counter = 1
        fs = 0
        X = []
        y = []
        # if selected_network == 'MLPClassifier' or selected_network == 'MLPKeras':
        #   window_size = {'tmin': -4.5, 'tmax': 5.0}
        #else:
        window_size = {'tmin': -0., 'tmax': 1}
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

        X_t, y_t = self.transform(X, y, selected_network)
        X_s, y_s = shuffle(X_t, y_t, random_state=0)
        splitted_values = self.splitData(X_s, y_s)

        return X, y, splitted_values

    def transform(self, X, y, selected_network):
        transform = Transformation()
        if selected_network == 'MLPClassifier' or selected_network == 'MLPKeras':
            X, y = transform.transformMLP(X, y)
        return X, y

    def splitData(self, X, y):
        train_idx = round(X.shape[0] * 0.8)
        test_idx = round(X.shape[0] * 0.9)

        X_train = X[0:train_idx, ]
        y_train = y[0:train_idx]
        X_validate = X[train_idx:test_idx, ]
        y_validate = y[train_idx:test_idx]
        X_test = X[test_idx:, ]
        y_test = y[test_idx:]

        splitted_values = {
            'X_train': X_train,
            'y_train': y_train,
            'X_validate': X_validate,
            'y_validate': y_validate,
            'X_test': X_test,
            'y_test': y_test
        }

        print("X.Shape", X.shape)
        print('X_train shape:', splitted_values['X_train'].shape)
        print('X_test shape:', splitted_values['X_test'].shape)
        print("y.Shape", y.shape)
        print('y_train shape:', splitted_values['y_train'].shape)
        print('y_test shape:', splitted_values['y_test'].shape)

        return splitted_values

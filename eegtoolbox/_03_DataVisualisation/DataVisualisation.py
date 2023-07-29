import numpy as np
from matplotlib import pyplot as plt


class DataVisualisation:

    def visualise(self, X, y, splitted_values, dataset_name):

        classes = np.unique(y)
        for c in classes:
            for i in range(X.shape[0]):
                if y[i] == c:
                    plt.figure()
                    plt.plot(X[i, 1])
                    plt.axis([0, X.shape[2], min(X[i, 1]), max(X[i, 1])])
                    plt.title('Event class: '+str(c))
                    plt.ylabel('amplitude')
                    plt.xlabel('datapoints')
                    plt.savefig('_03_DataVisualisation/EEG_Visualisation/'
                                + dataset_name + "_class_" + str(c))
                    break




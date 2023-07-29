import numpy as np

class Transformation:

    def transformMLP(self, X, y):

        print("X and y shapes before Transformation: ")
        print("X.shape: ", X.shape)
        print("y.shape: ", y.shape)

        x_transform = []
        y_transform = []
        counter = 1
        for i in range(X.shape[0]):
            for j in range(round(X.shape[1]/4)):
                if counter == 1:
                    x_transform = [X[i, j, :]]
                    y_transform = [y[i]]
                else:
                    x_transform = np.concatenate([x_transform, [X[i, j, :]]])
                    y_transform = np.concatenate([y_transform, [y[i]]])
            counter += 1

        print("X and y shapes after Transformation: ")
        print("X.shape: ", x_transform.shape)
        print("y.shape: ", y_transform.shape)

        return x_transform, y_transform
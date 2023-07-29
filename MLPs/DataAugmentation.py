import numpy as np
from sklearn.utils import shuffle

def count_and_print_tp_tn(y):
    count_tp_tn = [0, 0]
    for i in range(y.shape[0]):
        if y[i] == 0:
            count_tp_tn[0] += 1
        else:
            count_tp_tn[1] += 1
    print("Count TP: ", count_tp_tn[0])
    print("Count TN: ", count_tp_tn[1])
    return count_tp_tn

def augmentation(count_tp_tn, x_features_train, y_train_cls):
    # Data Augmentation
    while count_tp_tn[0] < count_tp_tn[1]:
        for n in range(y_train_cls.shape[0]):
            if y_train_cls[n] == 0:
                x_new_element = [x_features_train[n]]
                x_features_train = np.concatenate([x_features_train, x_new_element])
                y_train_cls = np.concatenate([y_train_cls, [0]])
                count_tp_tn[0] += 1
            if count_tp_tn[0] == count_tp_tn[1]:
                break

    shuffle(x_features_train, y_train_cls)
    return x_features_train, y_train_cls
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from datetime import datetime

class Evaluation:

    def plot(self, model_name, dataset_name, pred, y_test, history, num_epochs, acc):

        date = datetime.now().strftime("%Y_%m_%d__%H_%Mm_%Ss_")
        acc = str(round(acc,2))

        # Create Confusion Matrix
        cm = confusion_matrix(pred, y_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(model_name+": Confusion Matrix ; Test Accuracy: "+acc)
        plt.savefig('_06_Documentation/ConfusionMatrices/'+date+"_"
                    + model_name + "_" + dataset_name + '__confusionmatrix.png')


        # Accuracy Curve
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history["val_accuracy"])
        plt.axis([0, num_epochs, 0, 1])  # 1 = Accuracy_max
        plt.title(model_name + ": Accuracy Curve ; Test Accuracy: "+acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(["train", "val"], loc="best")
        plt.savefig('_06_Documentation/AccuracyCurves/'+ date + "_"
                        + model_name + "_" + dataset_name + '__accuracyCurve.png')

        # Loss Curve
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.axis([0, num_epochs, 0, 3])  # 3 = loss_max
        plt.title(model_name + ": Loss Curve ; Test Accuracy: "+acc)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('_06_Documentation/LossCurves/' + date + "_"
                        + model_name + "_" + dataset_name + '__lossCurve.png')


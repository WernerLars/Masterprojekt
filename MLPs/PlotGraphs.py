from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plotConfusionMatrix(model_name,cls_of_interest,pred,y_test_cls):
    cm = confusion_matrix(pred, y_test_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('figures/'+model_name+'_class_'+str(cls_of_interest)+'_confusionmatrix.png')


def plotGraphsSklearn(model_name,cls_of_interest, loss_curve):
    plt.figure()
    plt.plot(loss_curve)
    plt.savefig('figures/'+model_name+'_class_'+str(cls_of_interest)+'_loss.png')

def plotGraphsKeras(model_name,cls_of_interest,history,x_min,x_max,y_min,y_max):

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.axis([x_min,x_max,y_min,y_max])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('figures/'+model_name+'_class_'+str(cls_of_interest)+'_accuracy.png')
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.axis([x_min,x_max,y_min,y_max])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('figures/'+model_name+'_class_'+str(cls_of_interest)+'_loss.png')

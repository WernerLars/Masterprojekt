from _04_Networks.MLP.MLP import MLP
from _04_Networks.EEGNet.EEGNet_2a import EEGNet_2a
from _04_Networks.Transformer.Transformer_22Channels import Transformer_22Channels
from _04_Networks.FCN.FCN_22Channels import FCN_22Channels


class NetworkSelection:

    def selectNetwork(self, network_name, splitted_values):

        if network_name == 'EEGNet':
            network = EEGNet_2a()
            pred, y_test, history, num_epochs, acc = network.runEEGNet(splitted_values)
            return pred, y_test, history, num_epochs, acc

        elif network_name == 'MLPClassifier':
            network = MLP()
            pred, y_test, history, num_epochs, acc = network.runMLPClassifier(splitted_values)
            return pred, y_test, history, num_epochs, acc

        elif network_name == 'MLPKeras':
            network = MLP()
            pred, y_test, history, num_epochs, acc = network.runKerasModel(splitted_values)
            return pred, y_test, history, num_epochs, acc

        elif network_name == 'Transformer':
            network = Transformer_22Channels()
            pred, y_test, history, num_epochs, acc = network.runTransformer_22Channels(splitted_values)
            return pred, y_test, history, num_epochs, acc

        elif network_name == 'FCN':
            network = FCN_22Channels()
            pred, y_test, history, num_epochs, acc = network.runFCN_22Channels(splitted_values)
            return pred, y_test, history, num_epochs, acc

        ## Template zum Hinzufügen von Netzwerk (bitte vor Benutzung unten anhängen)
        # elif network_name == 'MyNewNetwork':
        #    network = NameOfPythonFile()
        #    pred, y_test, history, num_epochs = network.runFunctionForMyModel(splitted_values)
        #    return pred, y_test, history, num_epochs

        return [], [], [], [], []


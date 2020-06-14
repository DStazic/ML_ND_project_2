import torch
from torch import nn
import torch.nn.functional as func
from torchvision import models
from custom_exception import LayerRepresentationError, ModelRepresentationError


class Classifier(nn.Module):
    """
    wrapper for feedforward-only network
    -> backpropagation logic will be provided by pretrained model

    param input_size: number of features (int or list)
    param output_size: number of classes (int)
    param hidden_layers: size of each hidden layer (list)
    param dropout: dropout probability (float; default = 0.2)
    """

    def __init__(self, input_size, output_size, hidden_layers, dropout=0.2):
        super().__init__()
        self.input_size = [input_size] if isinstance(input_size, int) else input_size
        self.output_size = output_size
        input_all = self.input_size + hidden_layers
        self.hidden_layers = nn.ModuleList([nn.Linear(in_feat, out_feat)
                                            for in_feat, out_feat in zip(input_all[:-1], input_all[1:])])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        feed forward pass with each hidden layer output being relu activated and randomly
        ommiting a percentage (specified by dropout) of hidden units. Log-Softmax is used for
        activation of output layer results.
        """
        # flatten input vector
        x = x.view(x.shape[0], -1)
        for hl in self.hidden_layers:
            # linear combination and relu transformation of input feature
            x = func.relu(hl(x))
            x = self.dropout(x)
        x = self.output(x)
        return func.log_softmax(x, dim=1)


def get_classifier_representation(model):
    """
    Identifies how the fully connected layer (actual classifier) is represented within the pytorch model class.
    Returns str that can be used the access classifier in model

    :param model:
    :return:
    """
    if "classifier" in model.__dict__["_modules"]:
        classifier_representation = "classifier"
    elif "fc" in model.__dict__["_modules"]:
        classifier_representation = "fc"
    else:
        raise ModelRepresentationError
    return classifier_representation


def load_pretrained(arch):
    """
    Load pretrained model

    :param arch: None or str
    :return: torchvision model
    """

    model = [models.__dict__[k] for k in models.__dict__ if k == arch and callable(models.__dict__[k])]
    model = model[0](pretrained=True)
    return model


def build_model(arch, number_classes, hidden_units):
    """
    Load pretarained model and add custom fully-contacted layer

    :param arch:
    :param number_classes: int
    :param hidden_units: list of int
    :return:
    """
    # load pre-trained model
    model = load_pretrained(arch)
    # freeze weights of pre-trained model
    for param in model.parameters():
        param.requires_grad = False
    classifier_representation = get_classifier_representation(model)

    # noinspection PyShadowingNames
    def input_features(model, classifier_representation):
        classifier_type = type(model.__dict__["_modules"][classifier_representation])
        if classifier_type == torch.nn.modules.container.Sequential:
            features = model.__dict__["_modules"][classifier_representation][0].in_features
        elif classifier_type == torch.nn.modules.linear.Linear:
            features = model.__dict__["_modules"][classifier_representation].in_features
        else:
            raise LayerRepresentationError
        return features

    in_features = input_features(model, classifier_representation)
    classifier = Classifier(in_features, number_classes, hidden_units)
    model.__dict__["_modules"][classifier_representation] = classifier
    return model

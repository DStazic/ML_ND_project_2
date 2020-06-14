from termcolor import colored


class LayerRepresentationError(Exception):
    def __init__(self):
        self.msg = "Please check encoding of last layer in pretrained model"

    def __str__(self):
        return colored(repr(self.msg), "red")


class ModelRepresentationError(Exception):
    def __init__(self):
        self.msg = "Please check how classifier is encoded in pretrained model"

    def __str__(self):
        return colored(repr(self.msg), "red")


class ModelLoadError(Exception):
    def __init__(self):
        self.msg = "CUDA not supported... remap Tensors to CPU"

    def __str__(self):
        return colored(repr(self.msg), "red")

from torchvision import datasets, transforms
import torch
import os
import numpy as np
from termcolor import colored
import json
import zipfile
from model import Classifier, load_pretrained, get_classifier_representation
from custom_exception import ModelLoadError, CheckpointDirectoryPathError

# default_path = os.path.join(os.getcwd(), "flowers")
default_path_data = os.path.abspath("flowers")
default_path_checkpoint = os.path.abspath("checkpoint_iteration_7.pth")


def load_data(path):
    """
    load data and return

    :param path:
    :return:
    """

    if os.path.isdir(path):
        data_dir = path
    elif zipfile.is_zipfile(path + ".zip"):
        with zipfile.ZipFile(path + ".zip", 'r') as dir_in:
            dir_in.extractall(os.getcwd())
            data_dir = path
    else:
        raise FileNotFoundError(f'{colored(path,"red")} not referring to readable directory or zipfile\n'
                                f'use default path {colored(default_path_data, "red")}')

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # noinspection PyUnresolvedReferences
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # noinspection PyUnresolvedReferences
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    # noinspection PyUnresolvedReferences
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    print("successfully loaded data")
    return {"train": train_dataloader,
            "validation": valid_dataloader,
            "test": test_dataloader}


def load_class_mapping(file_name):
    """
    Load dictionary mapping idx to class names
    :param file_name: str
    :return:
    """
    path = os.path.abspath(file_name)
    with open(path, "r") as f_in:
        cat_to_name = json.load(f_in)
    cat_to_name = {int(k): v for k, v in cat_to_name.items()}
    return cat_to_name


def load_class_idx_correction(file_name):
    """
    Load correction dictionary for mapping predicted class idx to class names
    :param file_name:
    :return:
    """
    path = os.path.abspath(file_name)
    with open(path, "r") as f_in:
        class_idx_correction = json.load(f_in)
    class_idx_correction = {int(k): v for k, v in class_idx_correction.items()}
    return class_idx_correction


def save_checkpoint(model, arch, epochs, class_mapping, optimizer, save_dir):
    """
    Saves model checkpoint

    :param model: torch model class
    :param arch: str (pre-trained model architecture name)
    :param epochs: int (number of epochs)
    :param class_mapping: dict (dict for mapping class index to class names)
    :param optimizer: dict (optimizer state dict)
    :param save_dir: str (string specifying folder/file_name for checkpoint saving)
    :return:
    """

    classifier_representation = get_classifier_representation(model)
    classifier = model.__dict__["_modules"][classifier_representation]

    checkpoint = {'input_size': classifier.input_size,
                  'output_size': classifier.output_size,
                  'hidden_layers': [hl.out_features for hl in classifier.hidden_layers],
                  'state_dict': model.state_dict(),
                  'arch': arch,
                  'epochs': epochs,
                  'class_to_idx': class_mapping,
                  'optimizer_state': optimizer
                  }
    print("SAVE DIR", save_dir, type(save_dir))
    if not save_dir:
        file_name = "checkpoint.pth"
        print(f"no save directory provided. will save checkpoint ({file_name}) in current working dir")
    else:
        if "/" not in save_dir:
            raise CheckpointDirectoryPathError
        file_name = os.path.join(os.getcwd(), save_dir)
        save_dir_path = os.path.dirname(file_name)
        if not os.path.isdir(save_dir_path):
            os.makedirs(save_dir_path)

    torch.save(checkpoint, file_name)


def load_checkpoint(filepath, gpu=False, arch=None):
    """
    Loads checkpoint and returns model representation according to checkpoint information

    :param filepath: str (path indicating location of checkpoint)
    :param gpu: bool (specifies if GPU should be used for Tensor mapping)
    :param arch: str (name of model architecture used as pre-trained model)
    :return: trained model and corresponding checkpoint dictionary
    """

    device = torch.device("cuda" if gpu else "cpu")
    try:
        model_checkpoint = torch.load(filepath, map_location=device.type)
    except AssertionError:
        raise ModelLoadError

    # if no information on pre-trained model architecture:
    # 1. try to load from checkpoint
    # 2. if no information in checkpoint dict, used default checkpoint (vgg11, 3 hl [512, 256, 128])
    if arch:
        arch = arch
        print(f"loading specified architecture ... {arch}")
    else:
        try:
            arch = model_checkpoint["arch"]
            print(f"loading architecture used from checkpoint dictionary ... {arch}")
        except KeyError:
            print(colored(f"NO ARCHITECTURE INFO PROVIDED AND NO INFORMATION IN CHECKPOINT... "
                          f"load default (vgg11, 3 hl [512, 256, 128] ) {default_path_checkpoint}", "red"))
            arch = "vgg11"

            model_checkpoint = torch.load(default_path_checkpoint, map_location=device.type)

    # load used pre-trained model and freeze weights of pre-trained model
    # -> as previously, freeze weights of pre-trained model if further training of fc layers intended
    model = load_pretrained(arch)
    for param in model.parameters():
        param.requires_grad = False

    # add fc network and load trained weights
    model.classifier = Classifier(model_checkpoint["input_size"],
                                  model_checkpoint["output_size"],
                                  model_checkpoint["hidden_layers"])

    model.load_state_dict(model_checkpoint["state_dict"])
    model.class_to_idx = model_checkpoint["class_to_idx"]
    return model, model_checkpoint


def process_image(image, resize=255, crop_size=224):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array

    :param image: loaded image file
    :param resize: int
    :param crop_size: int
    :return:
    """
    width, height = image.size
    aspect_ratio = max(width, height) / min(width, height)
    size = (resize, int(resize * aspect_ratio)) if np.argmin(image.size) == 0 else (int(resize * aspect_ratio), resize)
    image = image.resize(size)
    # resized image
    width, height = image.size

    # define centered image format
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    image = image.crop((left, top, right, bottom))
    image = np.array(image)
    # standarize values to be within 0-1 range
    image_standarized = image / 255
    # normalize
    mean_norm_factors = [0.485, 0.456, 0.406]
    std_norm_factors = [0.229, 0.224, 0.225]
    image_normalized = (image_standarized - mean_norm_factors) / std_norm_factors
    # specify order of axes; otherwise image will be rotated by 90 deg
    image_normalized = image_normalized.transpose(2, 0, 1)
    return image_normalized

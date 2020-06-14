from PIL import Image
import torch
import numpy as np
from argparse import ArgumentParser
from utilities import process_image, load_checkpoint, load_class_idx_correction, load_class_mapping


# noinspection PyShadowingNames
def predict(image_path, model, topk):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    :param image_path:
    :param model:
    :param topk:
    :return:
    """

    image = Image.open(image_path)
    image = process_image(image)
    image_tensor = torch.from_numpy(image)
    # refactor tensor shape to cope with model specification
    image_tensor = image_tensor.view(1, *image_tensor.shape)
    image_tensor = image_tensor.type(torch.FloatTensor)
    with torch.no_grad():
        model.eval()

        output = model.forward(image_tensor)
        probs = torch.exp(output)
        class_probs, class_idx = probs.topk(topk, dim=1)
        return class_probs, class_idx


# noinspection PyShadowingNames
def show_class_name(class_idx, class_probs, category_names_path=None):
    """
    Display predicted class probability and either corresponding predicted class index or class name

    :param class_idx: list
    :param class_probs: list
    :param category_names_path: str
    :return:
    """
    # use correction mapping to properly remap predicted class index
    class_idx_correction = load_class_idx_correction("class_idx_correction.json")
    class_idx = [class_idx_correction[idx] for idx in class_idx.numpy().flatten()]
    class_probs = np.around(class_probs.numpy()[0], 2)

    if category_names_path:
        idx_to_class = load_class_mapping(category_names_path)
        print(f'category name: {str([idx_to_class[idx] for idx in class_idx]).strip("[]")}\n'
              f'class probability: {str(class_probs).strip("[]")}')

    else:
        print(f'class index: {str(class_idx).strip("[]")}\n'
              f'class probability: {str(class_probs).strip("[]")}')


if __name__ == "__main__":
    parser = ArgumentParser(__file__, description='make prediction using a trained network')
    parser.add_argument("image_path", type=str, help="path specifying location of input image")
    parser.add_argument("checkpoint", type=str, help="path specifying location of model checkpoint")
    parser.add_argument("--gpu", "-g", action="store_true", default=False)
    parser.add_argument("--arch", "-a", type=str, default=None)
    parser.add_argument("--top_k", "-c", type=int, default=1)
    parser.add_argument("--category_names", "-m", type=str, default=None)
    args = parser.parse_args()

    model, model_checkpoint = load_checkpoint(args.checkpoint, args.gpu, args.arch)
    class_probs, class_idx = predict(args.image_path, model, args.top_k)
    show_class_name(class_idx, class_probs, args.category_names)

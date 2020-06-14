from model import build_model
import torch
from torch import nn, optim
from utilities import load_data, load_class_mapping, save_checkpoint
from argparse import ArgumentParser


# noinspection PyShadowingNames
def apply_model(model, device, criterion, optimizer, train_dataloader, valid_dataloader, epochs=1, **options):
    """
    param training: bool (specifies if model should be trained)
    """

    model.to(device)

    # noinspection PyShadowingNames
    def evaluate(imageloader, model, device, criterion):
        """
        param imageloader: pytorch dataloader
        param model: pytorch model representation
        """
        nonlocal eval_loss
        nonlocal eval_accuracy

        # configure model for evaluation:
        # 1. ignore keep tracking of functions along the feedforward pass
        # 2. inactivate dropout
        with torch.no_grad():
            model.eval()
            for image_batch, label_batch in imageloader:
                image_batch = image_batch[:2]
                label_batch = label_batch[:2]
                # set proper configuration
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                # do forward pass and calculate loss
                output = model.forward(image_batch)
                eval_loss += criterion(output, label_batch)
                # convert model output to class probabilities
                class_prob = torch.exp(output)
                # calculate accuracy for validation set
                # 1. compare predicted class to true class for each batch
                # --> reshape label tensor to match shape of class_prediction tensor
                # 2. accumulate number of correctly predicted classes
                prob, top_class = class_prob.topk(1, dim=1)
                prediction = top_class == label_batch.view(*top_class.shape)
                eval_accuracy += torch.mean(prediction.type(torch.FloatTensor))

        # reset model configuration to training mode (dropout used for training)
        model.train()

    train_loss_by_epoch = []
    valid_loss_by_epoch = []
    valid_accuracy_by_epoch = []

    print("Training and validating model" if not options.get("only_validate") else "Only validation activated")
    for e in range(epochs):
        # iterate over all batches in training data and train classifier
        train_loss = 0
        eval_loss = 0
        eval_accuracy = 0

        if options.get("only_validate"):
            imageloader = options.get("only_validate")
            evaluate(imageloader, model, device, criterion)
            # noinspection PyUnresolvedReferences
            avg_test_loss = round((eval_loss / len(imageloader)).item(), 3)
            # noinspection PyUnresolvedReferences
            avg_test_accuracy = round((eval_accuracy / len(imageloader)).item(), 3)
            print(f"Test loss: {avg_test_loss}\n"
                  f"Test accuracy: {avg_test_accuracy}")
            return

        for idx, (image_batch, label_batch) in enumerate(train_dataloader):
            # set proper configuration
            image_batch = image_batch[:2]
            label_batch = label_batch[:2]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            # reset gradient to zero to avoid gradient accumulation
            # for all subsequent forward pass/backpropagation steps
            optimizer.zero_grad()

            # do forward pass and calculate loss
            output = model.forward(image_batch)
            loss = criterion(output, label_batch)

            # do backpropagation
            loss.backward()

            # update weights
            optimizer.step()

            # accumulate avg batch loss
            train_loss += loss

        # evaluate model using the appropriate evaluation set
        # 1. validation set for different models
        # 2. test set for generalization
        evaluate(valid_dataloader, model, device, criterion)

        # keep track of avg loss/accuracy by epoch
        avg_train_loss_by_epoch = round((train_loss / len(train_dataloader)).item(), 3)
        # noinspection PyUnresolvedReferences
        avg_test_loss_by_epoch = round((eval_loss / len(valid_dataloader)).item(), 3)
        # noinspection PyUnresolvedReferences
        avg_test_accuracy_by_epoch = round((eval_accuracy / len(valid_dataloader)).item(), 3)

        train_loss_by_epoch.append(avg_train_loss_by_epoch)
        valid_loss_by_epoch.append(avg_test_loss_by_epoch)
        valid_accuracy_by_epoch.append(avg_test_accuracy_by_epoch)

        msg = f"Epoch: {e + 1}\n\
        Train Loss (avg): {avg_train_loss_by_epoch}\n\
        Test Loss (avg): {avg_test_loss_by_epoch}\n\
        Test Accuracy (avg) {avg_test_accuracy_by_epoch}\n\
        -------------------"
        print(msg)

    metrics = {"train_loss_by_epoch": train_loss_by_epoch,
               "valid_loss_by_epoch": valid_loss_by_epoch,
               "valid_accuracy_by_epoch": valid_accuracy_by_epoch}

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(__file__, description="train network")
    parser.add_argument('path', type=str, help="path specifying dataset location")
    parser.add_argument("--arch", "-a", type=str, default="vgg11")
    parser.add_argument("--learning_rate", "-lr", type=int, default=0.003)
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--hidden_units", "-u", nargs="*", type=int,
                        help="specify each layer size like this (e.g 3 layers): 512 256 256")
    parser.add_argument("--gpu", "-g", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", "-p", type=str, default="checkpoint.pth")

    args = parser.parse_args()
    data = load_data(args.path)
    exit()
    idx_to_class = load_class_mapping("cat_to_name.json")
    model = build_model(args.arch, len(idx_to_class), args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = "cuda" if args.gpu else "cpu"
    metrics = apply_model(model, device, criterion, optimizer, data["train"], data["validation"], epochs=args.epochs)
    save_checkpoint(model, args.checkpoint_path,
                    arch=args.arch, epochs=args.epochs,
                    class_mapping=idx_to_class, optimizer=optimizer.state_dict())

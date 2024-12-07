import torch
import torchvision
from torch import nn


import utils
from utils import accuracy_fn


device = "cpu"

# Main code
def main():
    device = "cpu"

    model = torchvision.models.efficientnet_b3(weights='DEFAULT').to(device)

    # Freeze the effnet layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = 66

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1536,
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    results = utils.train(model = model)

if __name__ == "__main__":
    main()
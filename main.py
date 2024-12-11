import torch
import torchvision
from torch import nn
from torchvision import transforms
import pathlib
from pathlib import Path

import utils
from utils import accuracy_fn
import data_setup

# Main code
def main():

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define the path to your image folder
    current_dir = pathlib.Path().resolve()
    data_dir = current_dir / "Cat_Breeds/images"
    train_dir = current_dir / Path("train")
    test_dir = current_dir / Path("test")
    
    utils.get_data_libraries(data_dir,
                             train_dir,
                             test_dir,
                             clean_data = True,
                             low_image_treshold = 200,
                             split_train_ratio = 0.8,
                             split_experimental_ratio = 0.01)

    #print(f"data_dir: {data_dir}, train_dir: {train_dir}, test_dir: {test_dir}")

    # Write transform for image
    manual_transform = transforms.Compose([
        # Resize the images to 224x224
        transforms.Resize(size=(224, 224)),
        # Flip the images randomly on the horizontal
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ])

    # Get pretrained efficientnet model
    model = torchvision.models.efficientnet_b3(weights='DEFAULT').to(device)
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT 
    auto_transform = weights.transforms()

    BATCH_SIZE = 32
    NUM_WORKERS = 1

    # Creating train and test dataloder, getting class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                                   test_dir,
                                                                                   auto_transform,
                                                                                   BATCH_SIZE,
                                                                                   NUM_WORKERS)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)


    # Freeze the effnet layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1536, # Check torch summary for this value
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    results = utils.train(model = model,
                          train_dataloader = train_dataloader,
                          test_dataloader = test_dataloader,
                          optimizer = optimizer,
                          accuracy_fn = accuracy_fn,
                          loss_fn = loss_fn,
                          epochs = 5)

if __name__ == "__main__":
    main()
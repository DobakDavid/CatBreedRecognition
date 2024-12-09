"""
File containing various utility functions for PyTorch model training.
""" 
import torch
from torch import nn
import os
from tqdm.auto import tqdm
from PIL import Image
import random

from pathlib import Path
import shutil




def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def accuracy_fn(y_true, y_pred):
  """Calculates accuracy between truth labels and predictions.

  Args:
    y_true (torch.Tensor): Truth labels for predictions.
    y_pred (torch.Tensor): Predictions to be compared to predictions.

  Returns:
    [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
  """
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc 

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
  """ Functionizing a common PyTorch train loop.

  Args:
    model: PyTorch model to train.
    data_loader: DataLoader, containing the training data.
    loss_fn: Loss function for training. 
    accuracy_fn: Accuracy function for classification.
    device: PyTorch device for calculation.
    
  Returns:
    Tuple containing train loss and train accuracy.

  Example usage:
    train_step(model = model_0,
              data_loader = train_dataloader,
              loss_fn = nn.CrossEntropyLoss(),
              accuracy_fn = accuracy_fn,
              device = "cpu")
  """

  # Initialize training loss and training accuracy
  train_loss, train_acc = 0, 0

  # Put model into training mode
  model.train()
  for batch, (X, y) in enumerate(data_loader):

    # 0. Sending the data to device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    y_pred = model(X)

    # 2. Calculate the loss and accuracy
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim = 1))

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Divide total train loss and acc by length of train dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss: .5f} | Train acc: {train_acc:.2f}")

  return train_loss.item(), train_acc

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
  """ Functionizing a common PyTorch test loop.

  Args:
    model: PyTorch model to train.
    data_loader: DataLoader, containing the test data.
    loss_fn: Loss function for training. 
    accuracy_fn: Accuracy function for classification.
    device: PyTorch device for calculation.
    
  Returns:
    Tuple containing test loss and test accuracy.

  Example usage:
    test_step(model = model_0,
              data_loader = test_dataloader,
              loss_fn = nn.CrossEntropyLoss(),
              accuracy_fn = accuracy_fn,
              device = "cpu")
  """

  # Initialize test loss and training accuracy
  test_loss, test_acc = 0, 0
  model.eval()

  # Put model into inference mode
  with torch.inference_mode():
    for X_test, y_test in data_loader:

      # 0. Sending the data to device
      X_test, y_test = X_test.to(device), y_test.to(device)

      # 1. Forward pass
      test_pred = model(X_test)

      # 2. Calculate the loss and accuracy
      test_loss += loss_fn(test_pred, y_test)
      test_acc += accuracy_fn(y_true = y_test, y_pred = test_pred.argmax(dim = 1))

    # Divide total test loss by length of test dataloader
    test_loss /= len(data_loader)

    # Calculate the test acc average
    test_acc /= len(data_loader)

  # Return the results
  return test_loss.item(), test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          accuracy_fn,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

  """Trains PyTorch model.

  Args:
    model: PyTorch model to train.
    train_dataloader: DataLoader, containing the training data.
    test_dataloader: DataLoader, containing the test data.
    optimizer: Optimizer function for training. 
    accuracy_fn: Accuracy function for classification.
    loss_fn: Loss function for training. Default is nn.CrossEntropyLoss().
    epochs: Number of epochs, default is 5.
  
  Returns:
    Dictionary containing the name of the model, train loss, train accuracy, test loss, test accuracy.

  Example usage:
    train(model = model_0,
          train_dataloader = train_dataloader,
          test_dataloader = test_dataloader,
          optimizer = torch.optim.Adam(model.parameters(), lr=0.001),
          accuracy_fn = accuracy_fn,
          loss_fn = nn.CrossEntropyLoss(),
          epochs = 5) 
  """

  # 1. Create empty results dictionary
  results = {"model_name": [model.__class__.__name__],
             "train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []
             }

  # 2. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model = model,
                                       data_loader = train_dataloader,
                                       loss_fn = loss_fn,
                                       optimizer = optimizer,
                                       accuracy_fn = accuracy_fn,
                                       device = "cpu")
    test_loss, test_acc = test_step(model = model,
                                    data_loader = test_dataloader,
                                    loss_fn = loss_fn,
                                    accuracy_fn = accuracy_fn,
                                    device = "cpu")

    # 3. Print out the values
    print(f"Epoch: {epoch + 1} | "
          f"train_loss: {train_loss: .4f} | "
          f"train_acc: {train_acc: .4f} | "
          f"test_loss: {test_loss: .4f} | "
          f"test_acc: {test_acc: .4f}"
          )

    # 4. Update the results dictionary
    results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
    results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
    results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
    results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

  # Return the results at the end of the epochs
  return results

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = "cpu"):
  """Makes predictions for the data with the given model.

  Args:
    model: Trained PyTorch model.
    data: Input data represented as list for the model, the shape has to compatible with model's input.
    device: PyTorch device, the default is "cpu".

  Returns:
    Return the prediction probabilities for classes.

  Example usage:
    make_predictions(model=model_0,
                     data=Tulipe_example,
                     device=device)
  """
  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      # Prepare sample
      sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send to device

      # Forward pass
      pred_logit = model(sample)

      # Get prediction probability
      pred_prob = torch.softmax(pred_logit.squeeze(), dim = 0)

      # Get pred_prob off from device to cpu
      pred_probs.append(pred_prob.cpu())

  # Stack the pred_probs to turn list into a tensor
  return torch.stack(pred_probs)
  
def check_and_remove_corrupted_image(file_path):
  """Opens a file to check if it is a corrupted image. If the file is a corrupted image, the function removes it.

  Args:
    file_path: A file path to open.

  Returns:
    Returns True if the file was a corrupted image file.

  Example usage:
    check_and_remove_corrupted_image(model="../content/images/Tulipe_148461.jpg")
  """
  try:
    with Image.open(file_path) as img:
      img.load()  # Check if the image is corrupted (Use load instead of verify, because verify sometimes can be inaccurate)
    return False  # Image is not corrupted
  except (IOError, SyntaxError) as e:
    print(f"Removing corrupted image: {file_path} - {e}")
    os.remove(file_path)  # Remove corrupted image file
    return True  # Image was corrupted and removed

def scan_and_clean_directory(directory):
  """Walks through a directory and it's subdirectories iteratively, and removes all of the corrupted image files.

  Args:
    directory: A file path of a root directory to open.

  Example usage:
    scan_and_clean_directory(model="../content/images")
  """
  for root, dirs, files in os.walk(directory):
    for file in files:
      file_path = os.path.join(root, file)
      check_and_remove_corrupted_image(file_path)

def remove_low_image_classes(data_dir, treshold = 1):
  """Removes all of the image classes (subdirectories) in the given root directory.

  Args:
    data_dir: A file path of a root directory to open.
    treshold: The maximum value of images on the removed subdirectories.

  Example usage:
    remove_low_image_classes(data_dir = "../content/images",
                             treshold = 5)
  """
  for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
      num_images = len(os.listdir(class_path))
      if num_images <= treshold:  # Check if class has only one image or less
        shutil.rmtree(class_path)  # Remove the class folder
        print(f"Removed class with less than {treshold + 1} image(s): {class_name}") 

def split_data(data_dir: Path,
              train_dir: Path = Path("train"),
              test_dir: Path = Path("test"),
              train_ratio: float = 0.8,
              experimental_ratio: float = 0.5):
  """Splits image data into train and test directories.

  Args:
    data_dir: Path to the directory containing the image data.
    train_dir: Path to the directory where the training data will be saved.
    test_dir: Path to the directory where the testing data will be saved.
    train_ratio: The proportion of data to include in the training set.
    experimental_ratio: The proportion of the data to use from the original data set.

   Example usage:
      split_data(data_dir=model_0,
                 train_dir=Path("train"),
                 test_dir=Path("test"),
                 train_ratio = 0.8,
                 experimental_ratio = 0.5) 
  """

  # Remove the existing directories
  if train_dir.exists():
    if train_dir.is_dir(): # Check if the path is directory
      shutil.rmtree(train_dir) # Remove not empty directory
    else:
      train_dir.unlink()

  if test_dir.exists():
    if test_dir.is_dir(): # Check if the path is directory
      shutil.rmtree(test_dir) # Remove not empty directory
    else:
      test_dir.unlink()

  # Create train and test directories
  train_dir.mkdir(exist_ok=True)
  test_dir.mkdir(exist_ok=True)

  # Loop through each flower type subfolder
  for class_type in data_dir.iterdir():
      if class_type.is_dir():
        # Create corresponding subfolders in train and test directories
        (train_dir / class_type.name).mkdir(exist_ok=True)
        (test_dir / class_type.name).mkdir(exist_ok=True)

        # Get a list of all image files in the current flower type subfolder
        image_files = list(class_type.glob("*.jpg"))  # Adjust file extension if needed

        # Shuffle the image files randomly
        random.shuffle(image_files)

        # Calculate the number of images to use based on data_ratio
        num_images_to_use = int(len(image_files) * experimental_ratio)

        # Select a subset of images based on experimental_ratio
        image_files = image_files[:num_images_to_use]

        # Calculate the split index
        split_index = int(len(image_files) * train_ratio)

        # Copy images to train and test directories based on the split index
        for i, image_file in enumerate(image_files):
          if i < split_index:
            shutil.copy(image_file, train_dir / class_type.name / image_file.name)
          else:
            shutil.copy(image_file, test_dir / class_type.name / image_file.name)

def get_data_libraries(data_dir: Path,
                       train_dir: Path,
                       test_dir: Path,
                       clean_data: bool = True,
                       low_image_treshold: int = 1,
                       split_train_ratio: float = 0.8,
                       split_experimental_ratio: float = 0.8):
  """
  Docstring

  Args:
    data_dir:
    train_dir:
    test_dir:
    clean_data:
    low_image_treshold:
    split_train_ratio:
    split_experimental_ratio:
  
  Example usage:
    get_data_libraries(data_dir:
                       train_dir:
                       test_dir:
                       clead_data:
                       low_image_treshold:
                       split_train_ratio:
                       split_experimental_ratio:)
  """
  # Clean the dataset if required
  if clean_data:
    scan_and_clean_directory(data_dir)
    remove_low_image_classes(data_dir, low_image_treshold)
  
  # Split into trainining and test directory
  split_data(data_dir,
             train_dir,
             test_dir,
             split_train_ratio,
             split_experimental_ratio)




    



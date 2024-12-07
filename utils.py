"""
File containing various utility functions for PyTorch model training.
""" 
import torch
import os
from PIL import Image

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

def remove_single_image_classes(data_dir, treshold = 1):
  """Walks through a directory and it's subdirectories iteratively, and removes all of the corrupted image files.

  Args:
    directory: A file path of a root directory to open.

  Example usage:
    scan_and_clean_directory(model="../content/images")
  """
  for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
      num_images = len(os.listdir(class_path))
      if num_images <= treshold:  # Check if class has only one image or less
        shutil.rmtree(class_path)  # Remove the class folder
        print(f"Removed class with less than {treshold + 1} image(s): {class_name}") 

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
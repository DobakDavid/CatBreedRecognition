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
  """Checks the directory and all subdirectories for corrupted image files and removes them.

  Args:
    file_path: A target PyTorch model to save.
  
  Example usage:
    check_and_remove_corrupted_image(model="../content/images")
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
  for root, dirs, files in os.walk(directory):
    for file in files:
      file_path = os.path.join(root, file)
      check_and_remove_corrupted_image(file_path)
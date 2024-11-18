
# utils.py

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from typing import Dict

# Loss and metric functions
def dice_coeff_metric(pred, label):
    smooth = 1e-5
    intersection = 2. * (pred*label).sum() + smooth
    total_area = pred.sum() + label.sum() + smooth
    return intersection / total_area

def dice_coeff_loss(pred, label):
    smooth = 1e-5
    intersection = 2. * (pred*label).sum() + smooth
    total_area = pred.sum() + label.sum() + smooth
    return 1 - (intersection/total_area)

def bce_dice_loss(pred, label):
    dice_loss = dice_coeff_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss

def plot_result_history(model_history: Dict, epochs: int, save=False, save_to_folder: str=None):
    """
    Plot and Visualize the Dice Coefficient Metrics and Loss from Model.

    Args:
    ---------
    model_history  :  Dictionary of model's results (Training and Testing)
    epochs: iteration for model has been trained.
    save: Save image to destination folder.

    Return:
    ---------
    Plot and Visualization of Model's Performances
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    EPOCHS = range(epochs)
    axs[0].plot(EPOCHS, model_history["train_dice"], label="Train Dice")
    axs[0].plot(EPOCHS, model_history["test_dice"], label="Test Dice")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Dice")
    axs[0].set_title("Dice metric over epochs")
    
    axs[1].plot(EPOCHS, model_history["train_loss"], label="Train Loss")
    axs[1].plot(EPOCHS, model_history["test_loss"], label="Test Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss over epochs")
    
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()

    if save==True and save_to_folder is not None:
        fig.savefig(save_to_folder)
        print(f"Succesfull to save image to this directory: {save_to_folder}")

def save_model(model: torch.nn.Module,
               dest_dir: str=None):
    """
    Save model to destination folder we choose.

    Args:
    --------
    model: Model was built and trained on.
    dest_dir: Destination folder path for saving the model.

    Return:
    ---------
    Model is saved by using format '.pth'
    """

    model_weights = model.state_dict()
    torch.save(model_weights, dest_dir)
    print("Model has been succesfully saved")

def load_model(model: torch.nn.Module,
               path_model: str=None):
    """
    Load the model and return it for evaluating.
    
     Args:
    --------
    model: Model was built and trained on.
    path_model: Model path for saved model.

    Return:
    ---------
    Return the model with trained weights.
    """
    trained_weights_model = torch.load(path_model)
    model.load_state_dict(trained_weights_model)
    
    return model


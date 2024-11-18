# train.py

import os
import torch
from modulars import data_setup, engine, unet, utils
import argparse

# Define the parser
parser = argparse.ArgumentParser(description="Get some hyperparameters")

# Create an argument for num_epochs
parser.add_argument("--num_epochs",
                    type=int,
                    default=10,
                    help="The number of epochs to train for")

# Create an argument for batch_size
parser.add_argument("--batch_size",
                    type=int,
                    default=4,
                    help="The number of images per batch")

# Create an argument for learning rate
parser.add_argument("--learning_rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for training the model")

# Create an argument for test size
parser.add_argument("--test_size",
                    type=float,
                    default=0.2,
                    help="Constants to split the dataset")

# Create an argument for weight decay
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-2,
                    help="Constants for reducing overfit")

# Create an argument for root path
parser.add_argument("--root_path",
                    type=str,
                    default=r"C:\Users\Asus\Documents\Programming\Projects\brain-mri-segmentation",
                    help="Root path for accessing the dataset")


# Get our arguments from parser
args = parser.parse_args()

# Setup Hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
TEST_SIZE = args.test_size
WEIGHT_DECAY = args.weight_decay


# Setup root directory path (in the root path, should exist 'data' folder)
ROOT_PATH = args.root_path

# Setup target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# From data_setup, we create DataLoaders for training and testing
train_dataloaders, test_dataloaders = data_setup.create_dataloaders(root_path=ROOT_PATH,
                                                                    test_size=TEST_SIZE,
                                                                    batch_size=BATCH_SIZE,
                                                                    num_workers=0)
# Instantiate the model and optimizer
unet_model = unet.UNet(input_shape=3, output_shape=1)
optimizer = torch.optim.AdamW(params=unet_model.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       mode="min",
                                                       patience=10)

# Train the model and return results as dictionary
unet_results = engine.train_model(model=unet_model,
                                  train_dataloader=train_dataloaders,
                                  test_dataloader=test_dataloaders,
                                  loss_fn=utils.bce_dice_loss,
                                  dice_fn=utils.dice_coeff_metric,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  device=device,
                                  epochs=NUM_EPOCHS)

# Display the results for loss and dice metrics
utils.plot_result_history(model_history=unet_results,
                          epochs=NUM_EPOCHS,
                          save=True,
                          save_to_folder=os.path.join(ROOT_PATH, "results.png"))

# Save the model 
utils.save_model(model=unet_model,
                 dest_dir="unet_model.pth")

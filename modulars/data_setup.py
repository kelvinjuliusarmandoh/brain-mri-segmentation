
# Data_setup.py

import os
import pandas as pd
from glob import glob

from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from torch.utils.data import DataLoader
from modulars.dataset import define_transformer, BrainDataset

# CURRENT_WORKING_DIR = os.getcwd()
# BRAIN_MRI_ZIP = os.path.join(CURRENT_WORKING_DIR, "brain_mri_dataset.zip")
# DATA_DEST_FOLDER = os.path.join(CURRENT_WORKING_DIR, "data")

def extract_zip_data(path_to_mri_zip, path_to_dest_folder):
    if not os.path.isdir(path_to_dest_folder):
        os.mkdir(path_to_dest_folder)
    
    with ZipFile(path_to_mri_zip, mode='r') as zip_ref:
        zip_ref.extractall(path_to_dest_folder)
        
    print(f"Succesfully extract the data")


def create_dataframe(root_path):
    """
    Get the data through current working directory and data folder.

    Args:
    --------
    root_path: Current working directory 

    Return:
    --------
    Dataframe with 2 columns (images_path and masks_path).
    """
    
    mask_data = glob(os.path.join(root_path, "data", "kaggle_3m/*/*_mask.tif"))

    # Data storage
    images = []
    masks = []

    for data in mask_data:
        image = data.replace("_mask.tif", ".tif")
        mask = data
    
        # Append image and mask to list
        images.append(image)
        masks.append(mask)

    # Creating dataframe of images and masks for access easliy the data
    df = pd.DataFrame(data={
        "images_path": images,
        "masks_path": masks
    })

    return df

def split_the_dataset(df, test_size: int):
    image_train, image_test, mask_train, mask_test = train_test_split(df["images_path"], 
                                                                      df["masks_path"], 
                                                                      test_size=test_size, 
                                                                      random_state=42)
    

    train_df = pd.concat([image_train, mask_train], axis=1)
    test_df = pd.concat([image_test, mask_test], axis=1)
    
    print(f"Image train size: {len(image_train)} | Mask train size: {len(mask_train)}")
    print(f"Image test size: {len(image_test)} | Mask test size: {len(mask_test)}")
    
    return train_df, test_df
    

def create_dataloaders(root_path: str,
                       test_size: int,
                       batch_size,
                       num_workers):
    """
    Create Training and Testing Dataloaders.

    Get the DataFrame from extracting data in directory. Then, using Dataset to transform 
    the data and turn it into DataLoader by using some parameters.

    Args:
    --------
    root_path: Current working directory, where the data located.
    test_size: Size for splitting data in a dataframe
    batch_size: Size for batching the data.
    num_workers: Device is used.

    Return:
    ----------
    Training and Testing DataLoader.

    Example: 
    training_dataloader, test_dataloader = create_dataloaders(root_path=path_to_dir,
                                                              test_size=0.2,
                                                              batch_size=32,
                                                              num_workers=0)
    """
    # Create dataframe
    df = create_dataframe(root_path=root_path)

    # Split dataframe into train and test
    train_df, test_df = split_the_dataset(df,
                                          test_size=test_size)

    # Get the Transformers class from dataset.py
    train_transform, test_transform = define_transformer(augment=True)

    # Get the Dataset class from dataset.py
    train_dataset = BrainDataset(train_df,
                                 train_transform)
    test_dataset = BrainDataset(test_df,
                                test_transform)

    # Get and Return Train DataLoaders and Test DataLoaders
    train_dataloaders = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)
    
    test_dataloaders = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    return train_dataloaders, test_dataloaders
    
    

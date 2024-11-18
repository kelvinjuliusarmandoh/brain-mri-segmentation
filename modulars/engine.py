# engine.py

import torch
from tqdm.auto import tqdm
from timeit import default_timer as timer



## Train loop and test loop
def train_loop(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               dice_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Train the model for every epochs.

    Args:
    ---------

    model: Model for being trained
    dataloader: Data for the model while training
    loss_fn: Loss function for comparing ground truth label and
             prediction
    optimizer: Function for updating the parameter
    dice_fn: Dice Coeff metrics
    device: Device for training

    Returns:
    Model will be trained for some epochs.
    """
    model.to(device)
    model.train()
    train_loss, train_dice = 0, 0

    for batch, (image, mask) in enumerate(dataloader):
        image, true_mask = image.float().to(device), mask.float().to(device)

        y_pred_logits = model(image)
        y_pred_probs = torch.sigmoid(y_pred_logits) 
        y_pred_labels = torch.round(y_pred_probs)

        loss = loss_fn(y_pred_probs, true_mask)
        dice = dice_fn(y_pred_labels, true_mask)
        
        train_loss += loss
        train_dice += dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(dataloader)
    train_dice /= len(dataloader)

    return train_loss, train_dice

def test_loop(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             dice_fn,
             device: torch.device):
    """
    Testing the model and calculating the metrics.

    Args:
    ---------
    model: Model for being evaluated
    dataloader: Data for the model while training
    loss_fn: Loss function for comparing ground truth label and
             prediction
    iou_fn: Intersection over union function
    dice_fn: Dice Coeff metrics
    device: Device for training

    Returns:
    Model will be evaluated for some epochs.
    """
    
    model.to(device)
    model.eval()
    test_loss, test_dice = 0, 0
    with torch.inference_mode():
        for batch, (image, mask) in enumerate(dataloader):
            image, true_mask = image.float().to(device), mask.float().to(device)

            y_pred_logits = model(image)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_labels = torch.round(y_pred_probs)

            loss = loss_fn(y_pred_probs, true_mask)
            dice = dice_fn(y_pred_labels, true_mask)

            test_loss += loss
            test_dice += dice

        test_loss /= len(dataloader)
        test_dice /= len(dataloader)

    return test_loss, test_dice

## Train Function
def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                dice_fn,
                optimizer: torch.optim.Optimizer,
                scheduler,
                device: torch.device,
                epochs: int):
    
    results = {
        "train_dice": [],
        "train_loss": [],
        "test_dice": [],
        "test_loss": []
    }
    
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    start_timer = timer()
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-----------")

        train_loss, train_dice = train_loop(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            dice_fn=dice_fn,
                                            optimizer=optimizer,
                                            device=device)

        test_loss, test_dice = test_loop(model=model,
                                         dataloader=test_dataloader,
                                         loss_fn=loss_fn,
                                         dice_fn=dice_fn,
                                         device=device)

        end_timer = timer()
        print(f"Train loss: {train_loss:.3f} | Train Dice: {train_dice:.3f} | Test loss: {test_loss:.3f} | Test Dice: {test_dice:.3f}\n")

        # Applying scheduler
        scheduler.step(test_loss)

        # Append to dictionaries 
        results["train_dice"].append(train_dice.item())
        results["train_loss"].append(train_loss.item())
        results["test_dice"].append(test_dice.item())
        results["test_loss"].append(test_loss.item())
        
    # Timer calculation
    training_time = end_timer - start_timer
    print(f"Training time takes: {training_time:.3f}ms")
    
    return results

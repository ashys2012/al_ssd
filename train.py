from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR
)
from model import create_model
from custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR

import torch
import matplotlib.pyplot as plt
import time
import os
from new_model import create_dropout_model
import numpy as np
# torch.multiprocessing.set_sharing_strategy('file_system')

plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# Function for running training iterations.
def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets) 
        # print("loss_dict: ", loss_dict)
        # print("loss_dict.values(): ", loss_dict.values())
        # print("shape of loss_dict.values(): ", len(loss_dict.values()))

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary


#here we add the new validaiton loop with dropout enabled
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import sys

def enable_dropout(model):
    """Enable dropout layers during test-time for uncertainty estimation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def mc_dropout_ssd(data_loader, model, forward_passes):
    """Perform MC Dropout to estimate uncertainty in SSD model predictions."""
    all_predictions = []
    model.eval()
    enable_dropout(model)

    for _ in range(forward_passes):
        predictions = []
        for images, _ in data_loader:
            images = list(image.to(DEVICE) for image in images)
            with torch.no_grad():
                outputs = model(images)
            for output in outputs:
                if 'scores' in output:
                    probs = output['scores'].cpu().numpy()
                    predictions.append(probs)

        predictions = np.vstack(predictions)
        all_predictions.append(predictions)

    # Convert list to NumPy array for easier manipulation
    all_predictions = np.array(all_predictions)

    # Calculate mean probability across forward passes
    mean_probs = np.mean(all_predictions, axis=0)

    # Calculate entropy
    epsilon = sys.float_info.min
    entropy_per_prediction = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)

    # To average entropy over predictions for each image if needed
    entropy_per_image = np.split(entropy_per_prediction, len(data_loader.dataset))
    average_entropy_per_image = [np.mean(entropy) for entropy in entropy_per_image]

    return average_entropy_per_image



def set_dropout_mode(model, mode):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if mode == 'train':
                module.train()
            else:
                module.eval()


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    train_dataset = create_train_dataset(TRAIN_DIR)
    valid_dataset = create_valid_dataset(VALID_DIR)
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device.
    model = create_dropout_model(num_classes=NUM_CLASSES, size=RESIZE_TO, dropout_rate=0.3)
    model = model.to(DEVICE)
    #print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.0005, momentum=0.9, nesterov=True
    )
    scheduler = MultiStepLR(
        optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True
    )

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Mame to save the trained model with.
    MODEL_NAME = 'model'

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # To save best model.
    save_best_model = SaveBestModel()

    # Training loop.
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        entropy = mc_dropout_ssd(valid_loader, model, forward_passes=10)
    

        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")
        print("Entropy of predictions:", entropy)
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # save the best model till now.
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
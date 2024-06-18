#this is mc dropout code which works but is not correct I guess
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

def set_dropout_mode(model, mode):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if mode == 'train':
                module.train()
            else:
                module.eval()

def validate_with_mc_dropout(valid_data_loader, model, num_passes=10):
    print('Validating with MC Dropout')
    
    # Set the model to evaluation mode but with dropout layers active
    model.eval()
    set_dropout_mode(model, 'train')
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    
    for data in prog_bar:
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        all_outputs = []
        
        for _ in range(num_passes):
            with torch.no_grad():
                outputs = model(images)
                all_outputs.append(outputs)
        
        # Debugging: Check the structure of outputs
        print(f"Sample output structure: {all_outputs[0]}")
        
        # Calculate mean and variance for regression heads (bbox)
        bbox_means = []
        bbox_vars = []
        
        for i in range(len(images)):
            bbox_preds = [output[i]['boxes'].detach().cpu().numpy() for output in all_outputs]
            bbox_mean = np.mean(bbox_preds, axis=0)
            bbox_var = np.var(bbox_preds, axis=0)
            bbox_means.append(bbox_mean)
            bbox_vars.append(bbox_var)
        
        # Calculate entropy for classification heads
        cls_entropies = []
        
        for i in range(len(images)):
            cls_preds = [output[i]['scores'].detach().cpu().numpy() for output in all_outputs]
            cls_preds_stacked = np.stack(cls_preds, axis=0)  # Stack predictions
            cls_entropy = -np.mean(np.sum(cls_preds_stacked * np.log(cls_preds_stacked + 1e-10), axis=-1), axis=0)
            cls_entropies.append(cls_entropy)
        
        # Ensure cls_entropies has the correct shape
        for i in range(len(images)):
            num_boxes = bbox_means[i].shape[0]
            if len(cls_entropies[i].shape) == 0:
                cls_entropies[i] = np.full((num_boxes,), cls_entropies[i])
        
        # For mAP calculation using Torchmetrics.
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = torch.tensor(bbox_means[i])
            preds_dict['scores'] = torch.tensor(cls_entropies[i])
            preds_dict['labels'] = all_outputs[0][i]['labels'].detach().cpu()  # Assuming labels remain consistent
            
            # Debugging: Check shapes
            print(f"true_dict['boxes'].shape: {true_dict['boxes'].shape}")
            print(f"true_dict['labels'].shape: {true_dict['labels'].shape}")
            print(f"preds_dict['boxes'].shape: {preds_dict['boxes'].shape}")
            print(f"preds_dict['scores'].shape: {preds_dict['scores'].shape}")
            print(f"preds_dict['labels'].shape: {preds_dict['labels'].shape}")
            
            preds.append(preds_dict)
            target.append(true_dict)
    
    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    
    # Reset the model to its original state
    set_dropout_mode(model, 'eval')
    
    return metric_summary, bbox_vars, cls_entropies
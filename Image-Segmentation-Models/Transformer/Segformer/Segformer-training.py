import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from PIL import Image
from transformers import SegformerImageProcessor
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import hf_hub_download
import albumentations as A
from skimage import exposure
from torchvision.transforms.functional import to_tensor
from torch.optim import AdamW
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import sys
from torchmetrics import JaccardIndex
import os
import shutil


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images and masks.
            transform (callable, optional): A function/transform that takes in an image and mask and returns transformed versions.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "masks")

        # Gather image and annotation file names
        self.images = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])
        self.annotations = sorted([f for f in os.listdir(self.ann_dir) if os.path.isfile(os.path.join(self.ann_dir, f))])

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)



    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')

        mask_path = os.path.join(self.ann_dir, self.annotations[idx])
        mask = Image.open(mask_path).convert('L')

        # Convert PIL images to numpy arrays for albumentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']

        # Convert numpy arrays back to tensors
        image_tensor = to_tensor(image_np)
        mask_tensor = torch.from_numpy(mask_np).long()

        encoded_inputs = {"pixel_values": image_tensor, "labels": mask_tensor}

        return encoded_inputs

# Custom CLAHE function using skimage
def apply_clahe(image, clip_limit=0.5, **kwargs):
    image_float = image.astype(np.float32)

    # Apply CLAHE
    image_clahe = exposure.equalize_adapthist(image_float, clip_limit=clip_limit)

    # Return the image in float32 and normalized range [0, 1]
    return image_clahe.astype(np.float32)

# DeepSea Transforms
transforms_deepsea = A.ReplayCompose([
    A.ToFloat(max_value=255.0),
    A.Resize(height=512, width=512),
    A.Lambda(image=apply_clahe, name='apply_clahe'),
    A.GaussNoise(var_limit=(0.005, 0.005), p=1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
])


# Instantiate the training dataset with corrected paths
train_dataset = SemanticSegmentationDataset(
    root_dir='/data/leuven/360/vsc36057/images-masks-full-black-png',
    transform = transforms_deepsea
)


# Assuming you have an array of indices to split, e.g., np.arange(len(train_dataset))
indices = np.arange(len(train_dataset))

# Split indices into train and validation sets
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Creating PyTorch datasets for train and validation, ensuring transforms are applied
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

# Create dataloaders for train and validation sets with the transforms_deepsea applied
train_dataloader = DataLoader(train_subset, batch_size=4, num_workers=2, shuffle=True)
valid_dataloader = DataLoader(val_subset, batch_size=4, num_workers=2, shuffle=False)


# Define your custom id2label and label2id mappings
id2label = {0: 'background', 1: 'cell'}  # Replace 'background' and 'object' with your actual class names
label2id = {v: k for k, v in id2label.items()}




# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=2,
                                                         id2label=id2label,
                                                         label2id=label2id,
                                                         cache_dir="/data/leuven/360/vsc36057/cache")



metric = evaluate.load("mean_iou")


# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=0.00006)

jaccard_index = JaccardIndex(task="binary", num_classes=len(id2label)).to(device)

# Prepare a Pandas DataFrame to store the metrics for each epoch
metrics_df = pd.DataFrame()

# Training loop
for epoch in range(400):  # loop over the dataset multiple times
    # Temporary lists to store metrics for each epoch
    train_loss_list = []
    train_mean_iou_list = []
    train_mean_accuracy_list = []
    train_jaccard_index_list = []

    val_loss_list = []
    val_mean_iou_list = []
    val_mean_accuracy_list = []
    val_jaccard_index_list = []

    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{200} - Train")):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()

        # Evaluate with torchmetrics
        with torch.no_grad():
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # Update evaluate metric
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            # Update torchmetrics Jaccard Index
            jaccard_index.update(predicted, labels)

        # Add metrics to the list after each batch
        train_loss_list.append(loss.item())

        if idx % 100 == 0 or idx == len(train_dataloader) - 1:
            # Compute evaluate metrics
            computed_metrics = metric._compute(
                predictions=predicted.cpu(),
                references=labels.cpu(),
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=False,
            )

            # Compute Jaccard Index
            current_jaccard_index = jaccard_index.compute()
            jaccard_index.reset()  # Reset after each logging step

            # Append metrics to the lists
            train_mean_iou_list.append(computed_metrics["mean_iou"])
            train_mean_accuracy_list.append(computed_metrics["mean_accuracy"])
            train_jaccard_index_list.append(current_jaccard_index.item())

    # Calculate mean of the train metrics lists
    train_mean_loss = np.mean(train_loss_list)
    train_mean_iou = np.mean(train_mean_iou_list)
    train_mean_accuracy = np.mean(train_mean_accuracy_list)
    train_mean_jaccard_index = np.mean(train_jaccard_index_list)

    # Validation loop
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{200} - Validation")):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            # Evaluate with torchmetrics
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # Update evaluate metric
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            # Update torchmetrics Jaccard Index
            jaccard_index.update(predicted, labels)

            # Add metrics to the list after each batch
            val_loss_list.append(loss.item())

            if idx % 100 == 0 or idx == len(valid_dataloader) - 1:
                # Compute evaluate metrics
                computed_metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=labels.cpu(),
                    num_labels=len(id2label),
                    ignore_index=255,
                    reduce_labels=False,
                )

                # Compute Jaccard Index
                current_jaccard_index = jaccard_index.compute()
                jaccard_index.reset()  # Reset after each logging step

                # Append metrics to the lists
                val_mean_iou_list.append(computed_metrics["mean_iou"])
                val_mean_accuracy_list.append(computed_metrics["mean_accuracy"])
                val_jaccard_index_list.append(current_jaccard_index.item())

    # Calculate mean of the validation metrics lists
    val_mean_loss = np.mean(val_loss_list)
    val_mean_iou = np.mean(val_mean_iou_list)
    val_mean_accuracy = np.mean(val_mean_accuracy_list)
    val_mean_jaccard_index = np.mean(val_jaccard_index_list)

    # Print metrics after each epoch
    print(f"Epoch {epoch+1} metrics:")
    print(f"Train Loss: {train_mean_loss}")
    print(f"Train Mean IoU: {train_mean_iou}")
    print(f"Train Mean Accuracy: {train_mean_accuracy}")
    print(f"Train Jaccard Index: {train_mean_jaccard_index}")
    print(f"Validation Loss: {val_mean_loss}")
    print(f"Validation Mean IoU: {val_mean_iou}")
    print(f"Validation Mean Accuracy: {val_mean_accuracy}")
    print(f"Validation Jaccard Index: {val_mean_jaccard_index}")

    # Append to the metrics DataFrame
    epoch_metrics = {
        "Epoch": epoch + 1,
        "Train Loss": train_mean_loss,
        "Train Mean IoU": train_mean_iou,
        "Train Mean Accuracy": train_mean_accuracy,
        "Train Jaccard Index": train_mean_jaccard_index,
        "Validation Loss": val_mean_loss,
        "Validation Mean IoU": val_mean_iou,
        "Validation Mean Accuracy": val_mean_accuracy,
        "Validation Jaccard Index": val_mean_jaccard_index
    }
    epoch_metrics_df = pd.DataFrame(epoch_metrics, index=[0])
    metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)

    # Save model at the end of each epoch
    model_save_name = f'segmentation_model_epoch_{epoch+1}_mit-b0.pth'
    path = f"/data/leuven/360/vsc36057/Training weights + evaluation metrics/Segformer MIT B-0/{model_save_name}"
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

    # Save the metrics DataFrame to an Excel file after each epoch
    excel_path = f"/data/leuven/360/vsc36057/Training weights + evaluation metrics/Segformer MIT B-0/Excel Metrics/training_metrics_epoch_{epoch+1}_mit-b0.xlsx"
    metrics_df.to_excel(excel_path, index=False)
    print(f"Saved training metrics to Excel for epoch {epoch+1}.")

print("Training completed.")







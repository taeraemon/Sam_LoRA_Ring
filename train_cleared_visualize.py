# ------------------------------------------------------------------------------------------------
# Import necessary libraries
import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

# Import custom modules
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")
# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Main training loop
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

total_loss = []

for epoch in range(num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):
        
        outputs = model(batched_input=batch,
                        multimask_output=False)

        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
        loss = seg_loss(stk_out, stk_gt.float().to(device))
        
        # ------------------------------------------------------------------------------------------------
        # Visualize the output
        
        import numpy as np

        # 데이터 준비
        out_arr = stk_out[0, 0].detach().cpu().numpy()
        gt_arr = stk_gt[0, 0].detach().cpu().numpy()

        x = np.arange(out_arr.shape[1])
        y = np.arange(out_arr.shape[0])
        x, y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(12, 10))

        # stk_out 2D
        ax1 = fig.add_subplot(2, 2, 1)
        im1 = ax1.imshow(out_arr)
        ax1.set_title('stk_out 2D')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # stk_out 3D
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf1 = ax2.plot_surface(x, y, out_arr, cmap='viridis')
        ax2.set_title('stk_out 3D')
        fig.colorbar(surf1, ax=ax2, shrink=0.5, aspect=10)

        # stk_gt 2D
        ax3 = fig.add_subplot(2, 2, 3)
        im2 = ax3.imshow(gt_arr)
        ax3.set_title('stk_gt 2D')
        plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)

        # stk_gt 3D
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        surf2 = ax4.plot_surface(x, y, gt_arr, cmap='viridis')
        ax4.set_title('stk_gt 3D')
        fig.colorbar(surf2, ax=ax4, shrink=0.5, aspect=10)

        plt.tight_layout()
        plt.show()
        # ------------------------------------------------------------------------------------------------
        
        optimizer.zero_grad()
        loss.backward()
        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')
# ------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank_{rank}.safetensors")
# ------------------------------------------------------------------------------------------------



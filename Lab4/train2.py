import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import argparse
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

# Import your custom modules
from src.dataloader import get_dataloaders
from src.promptir_model import PromptIR
from src.utils import copy_source_code

# Perceptual loss (VGG-based)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision import models
        vgg = models.vgg16(pretrained=True).features.eval()
        self.vgg_layers = vgg
        self.layer_weights = {
            '3': 1.0,   # relu1_2
            '8': 1.0,   # relu2_2
            '15': 1.0,  # relu3_3
        }
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        loss = 0.0
        x_vgg, y_vgg = x, y
        
        for name, module in self.vgg_layers._modules.items():
            x_vgg = module(x_vgg)
            y_vgg = module(y_vgg)
            
            if name in self.layer_weights:
                loss += self.layer_weights[name] * F.mse_loss(x_vgg, y_vgg)
                
        return loss

# PSNR loss - maximizing PSNR is equivalent to minimizing MSE, but we'll use it directly
class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
        
    def forward(self, x, y):
        mse = F.mse_loss(x, y, reduction='mean')
        psnr_val = 10 * torch.log10(self.max_val**2 / mse)
        # Return negative PSNR because we want to maximize PSNR (minimize loss)
        return -psnr_val

# Combined loss function for image restoration
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, psnr_weight=0.1, ssim_weight=0.1, perceptual_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.psnr_loss = PSNRLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.l1_weight = l1_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        psnr = self.psnr_loss(pred, target)
        
        # SSIM loss: 1-SSIM because we want to minimize (SSIM is higher = better)
        ssim_val = ssim(pred, target, data_range=1.0)
        ssim_loss = 1.0 - ssim_val
        
        # Perceptual loss
        percep = self.perceptual_loss(pred, target)
        
        # Combine all losses
        total_loss = (self.l1_weight * l1 + 
                      self.psnr_weight * psnr + 
                      self.ssim_weight * ssim_loss + 
                      self.perceptual_weight * percep)
                      
        return total_loss, {
            'l1': l1,
            'psnr': psnr,
            'ssim': ssim_loss,
            'perceptual': percep
        }

class EnhancedPromptIRModel(pl.LightningModule):
    def __init__(self, learning_rate=2e-4, max_epochs=150, 
                 l1_weight=1.0, psnr_weight=0.1, ssim_weight=0.1, perceptual_weight=0.01,
                 rain_weight=1.0, snow_weight=1.0):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CombinedLoss(
            l1_weight=l1_weight, 
            psnr_weight=psnr_weight, 
            ssim_weight=ssim_weight,
            perceptual_weight=perceptual_weight
        )
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.rain_weight = rain_weight
        self.snow_weight = snow_weight
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        degraded_imgs = batch['degraded']
        clean_imgs = batch['clean']
        degradation_types = batch['degradation_type']
        
        # Forward pass
        restored = self(degraded_imgs)
        
        # Calculate loss with detailed components
        loss, loss_components = self.loss_fn(restored, clean_imgs)
        
        # Apply weights for different degradation types
        rain_mask = (degradation_types == 0)
        snow_mask = (degradation_types == 1)
        
        # Calculate loss separately for rain and snow
        if torch.any(rain_mask):
            rain_loss, _ = self.loss_fn(restored[rain_mask], clean_imgs[rain_mask])
        else:
            rain_loss = torch.tensor(0.0, device=self.device)
            
        if torch.any(snow_mask):
            snow_loss, _ = self.loss_fn(restored[snow_mask], clean_imgs[snow_mask])
        else:
            snow_loss = torch.tensor(0.0, device=self.device)
        
        # Apply degradation-specific weights
        weighted_loss = (
            (self.rain_weight * rain_loss * rain_mask.sum() + 
             self.snow_weight * snow_loss * snow_mask.sum()) / 
            (rain_mask.sum() + snow_mask.sum())
        )
        
        # Log all losses
        self.log("train_loss", weighted_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_l1_loss", loss_components['l1'], on_step=False, on_epoch=True, logger=True)
        self.log("train_psnr_loss", loss_components['psnr'], on_step=False, on_epoch=True, logger=True)
        self.log("train_ssim_loss", loss_components['ssim'], on_step=False, on_epoch=True, logger=True)
        self.log("train_percep_loss", loss_components['perceptual'], on_step=False, on_epoch=True, logger=True)
        
        if torch.any(rain_mask):
            self.log("train_rain_loss", rain_loss, on_step=False, on_epoch=True, logger=True)
        if torch.any(snow_mask):
            self.log("train_snow_loss", snow_loss, on_step=False, on_epoch=True, logger=True)
            
        # Log sample images (only once every 100 batches to save space)
        if batch_idx % 100 == 0:
            self._log_images(degraded_imgs, restored, clean_imgs, degradation_types, batch_idx)
            
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        degraded_imgs = batch['degraded']
        clean_imgs = batch['clean']
        degradation_types = batch['degradation_type']
        
        # Forward pass
        restored = self(degraded_imgs)
        
        # Calculate loss with detailed components
        loss, loss_components = self.loss_fn(restored, clean_imgs)
        
        # Calculate PSNR and SSIM
        # First unnormalize the images if needed
        def unnormalize(img):
            return img * torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1) + \
                   torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            
        restored_unorm = unnormalize(restored)
        clean_unorm = unnormalize(clean_imgs)
        
        # Calculate metrics for all images
        psnr_val = psnr(restored_unorm, clean_unorm, data_range=1.0)
        ssim_val = ssim(restored_unorm, clean_unorm, data_range=1.0)
        
        # Calculate metrics separately for rain and snow images
        rain_mask = (degradation_types == 0)
        snow_mask = (degradation_types == 1)
        
        # Rain metrics
        rain_psnr = psnr(restored_unorm[rain_mask], clean_unorm[rain_mask], data_range=1.0) if torch.any(rain_mask) else torch.tensor(0.0)
        rain_ssim = ssim(restored_unorm[rain_mask], clean_unorm[rain_mask], data_range=1.0) if torch.any(rain_mask) else torch.tensor(0.0)
        
        # Snow metrics
        snow_psnr = psnr(restored_unorm[snow_mask], clean_unorm[snow_mask], data_range=1.0) if torch.any(snow_mask) else torch.tensor(0.0)
        snow_ssim = ssim(restored_unorm[snow_mask], clean_unorm[snow_mask], data_range=1.0) if torch.any(snow_mask) else torch.tensor(0.0)
        
        # Log all metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_psnr", psnr_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ssim", ssim_val, on_step=False, on_epoch=True, logger=True)
        
        # Log component losses
        for name, value in loss_components.items():
            self.log(f"val_{name}_loss", value, on_step=False, on_epoch=True, logger=True)
        
        # Log degradation-specific metrics
        self.log("val_psnr_rain", rain_psnr, on_step=False, on_epoch=True, logger=True)
        self.log("val_psnr_snow", snow_psnr, on_step=False, on_epoch=True, logger=True)
        self.log("val_ssim_rain", rain_ssim, on_step=False, on_epoch=True, logger=True)
        self.log("val_ssim_snow", snow_ssim, on_step=False, on_epoch=True, logger=True)
        
        # Log sample images (only for the first batch to save space)
        if batch_idx == 0:
            self._log_images(degraded_imgs, restored, clean_imgs, degradation_types, batch_idx, prefix='val')
            
        return {"val_loss": loss, "val_psnr": psnr_val, "val_ssim": ssim_val}
    
    def _log_images(self, degraded, restored, clean, degradation_types, batch_idx, prefix='train'):
        # Convert tensors to images (unnormalize, convert to numpy)
        def tensor_to_img(tensor):
            img = tensor.clone().detach().cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = img.clamp(0, 1)
            img = img.numpy().transpose(1, 2, 0)
            return np.clip(img, 0, 1)
        
        # Log a few examples
        n_samples = min(4, len(degraded))
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))
        
        for i in range(n_samples):
            deg_type = "Rain" if degradation_types[i].item() == 0 else "Snow"
            
            # Original degraded image
            axes[i, 0].imshow(tensor_to_img(degraded[i]))
            axes[i, 0].set_title(f"Degraded ({deg_type})")
            axes[i, 0].axis('off')
            
            # Restored image
            axes[i, 1].imshow(tensor_to_img(restored[i]))
            axes[i, 1].set_title("Restored")
            axes[i, 1].axis('off')
            
            # Ground truth clean image
            axes[i, 2].imshow(tensor_to_img(clean[i]))
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Convert the figure to a tensor for logging
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Log the images
        self.logger.experiment.add_image(f'{prefix}_samples/batch_{batch_idx}', 
                                        np.transpose(img, (2, 0, 1)), 
                                        global_step=self.global_step)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Use OneCycleLR for better convergence
        steps_per_epoch = 800  # Approximate steps per epoch - adjust based on your dataset size
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=self.max_epochs,
            pct_start=0.1,  # Warm-up for 10% of training
            div_factor=25.0,  # Initial LR = max_lr/div_factor
            final_div_factor=10000.0  # Final LR = max_lr/(div_factor*final_div_factor)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

def main(args):
    # Create directories if they don't exist
    os.makedirs(args.result_dir, exist_ok=True)

    checkpoint_dir = os.path.join(args.result_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.ckpt_dir = checkpoint_dir
    log_dir = os.path.join(args.result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    # Set up logger
    nowtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger = TensorBoardLogger(save_dir=args.log_dir, name=f"{nowtime}")

    # Copy source code to the result directory
    copy_source_code('.', args.result_dir)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='promptir-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=3,  # Save the 3 best models based on validation PSNR
        monitor='val_psnr',
        mode='max',  # Higher PSNR is better
        save_last=True  # Also save the last model
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        val_split=0.1  # Use 10% of training data for validation
    )
    
    # Initialize the model
    model = EnhancedPromptIRModel(
        learning_rate=args.lr, 
        max_epochs=args.epochs,
        l1_weight=args.l1_weight,
        psnr_weight=args.psnr_weight,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        rain_weight=args.rain_weight,
        snow_weight=args.snow_weight
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_gpus if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )
    
    # Train the model
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print(f"Training completed. Best model checkpoint saved at: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Enhanced PromptIR model')
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--result_dir', type=str, default='./results/exp1', help='Directory to save results')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    
    # Loss function weights
    parser.add_argument('--l1_weight', type=float, default=1.0, help='Weight for L1 loss')
    parser.add_argument('--psnr_weight', type=float, default=0.1, help='Weight for PSNR loss')
    parser.add_argument('--ssim_weight', type=float, default=0.1, help='Weight for SSIM loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.01, help='Weight for perceptual loss')
    
    # Degradation-specific weights
    parser.add_argument('--rain_weight', type=float, default=1.0, help='Weight for rain degradation')
    parser.add_argument('--snow_weight', type=float, default=1.0, help='Weight for snow degradation')
    
    args = parser.parse_args()
    best_model_path = main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from pokemon_sprite_diffusion.utils.image_processing import batch_to_pil, tensor_to_pil
from typing import List, Tuple, Optional, Union
import os
from PIL import Image

def get_timestep_embedding(
    timesteps: torch.Tensor,
    emb_dim: int
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps (torch.Tensor): Diffusion timesteps of shape (batch_size,).
        emb_dim (int): Embedding dimension.
        
    Returns:
        torch.Tensor: Time step embeddings of shape (batch_size, embedding_dim)
    """
    half_dim = emb_dim // 2
    
    # Logarithmically spaced frequencies
    emb = math.log(10000) / (half_dim - 1)
    # Remember to move to the same device as timesteps!
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    
    # Multiply timesteps (as float) with exponents (broadcasting to shape (batch_size, half_dim))
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    
    # Apply sine to the first half and cosine to the second half, then concatenate
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    # Pad with zero if embedding_dim is odd
    if emb_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    
    return emb

class DiffusionTransformer(nn.Module):
    def __init__(
        self, 
        image_size: int,
        patch_size: int,
        in_channels: int,
        emb_dim: int,
        depth: int,
        nheads: int,
        mlp_ratio: float = 4.0,
        time_emb_dim: int = 128
    ) -> None:
        """
        Args:
            image_size (int): Height/width of the (square) image.
            patch_size (int): Height/width of the (square) patch.
            in_channels (int): Number of channels in the input image.
            emb_dim (int): Embedding dimension.
            depth (int): Number of transformer layers.
            nheads (int): Number of attention heads.
            mlp_ratio (float): Ratio for the hidden dimension in feed-forward layers.
            time_emb_dim (int): Time embedding dimension. 
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.depth = depth
        self.nheads = nheads
        self.mlp_ratio = mlp_ratio
        self.time_emb_dim = time_emb_dim
        
        # Convolution layer
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))

        # Time embedding: sinusoidal embedding followed by MLP.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nheads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            batch_first=True,
            norm_first=True
        )
        final_norm_layer = nn.LayerNorm(emb_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=final_norm_layer,
            enable_nested_tensor=False
        )

        # Unembedding layer
        self.head = nn.Linear(emb_dim, patch_size * patch_size * in_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Pass the input through the model to predict noise.
        
        Args:
            x (torch.Tensor): Noisy image of shape (batch_size, in_channels, image_size, image_size).
            t (torch.Tensor): Diffusion timesteps of shape (batch_size,).
            
        Returns:
            torch.Tensor: Predicted noise with shape (batch_size, in_channels, image_size, image_size).
        """
        batch_size = x.size(0)
        
        # Convert image to patch tokens.
        # x: (batch_size, in_channels, image_size, image_size) -> (batch_size, emb_dim, H', W'),
        # where H' = image_size // patch_size.
        x = self.patch_embed(x)
        # Flatten patches and transpose: (batch_size, emb_dim, H', W') -> (batch_size, emb_dim, num_patches) -> (batch_size, num_patches, emb_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embeddings.
        x = x + self.pos_embed
        
        # Add time embedding to each token.
        t_emb = get_timestep_embedding(t, self.time_emb_dim)  # shape: (batch_size, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # shape: (batch_size, emb_dim)
        t_emb = t_emb.unsqueeze(1)    # shape: (batch_size, 1, emb_dim), need to broadcast
        x = x + t_emb
        
        # Process tokens with transformer.
        # Reminder: PyTorch transformer usually expects (sequence_length, batch_size, emb_dim) but we set batch_first = True
        x = self.transformer(x)
        
        # Unembed tokens back to pixel space.
        # (batch_size, num_patches, emb_dim) -> (batch_size, num_patches, patch_size * patch_size * in_channels)
        x = self.head(x)
        
        # Reconstruct the image.
        H = W = self.image_size // self.patch_size
        # (batch_size, num_patches, patch_size * patch_size * in_channels) -> (batch_size, H, W, self.patch_size, self.patch_size, in_channels)
        x = x.view(batch_size, H, W, self.patch_size, self.patch_size, -1)
        # (batch_size, H, W, self.patch_size, self.patch_size, in_channels) -> (batch_size, in_channels, H, self.patch_size, W, self.patch_size)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # (batch_size, in_channels, H, self.patch_size, W, self.patch_size) -> (batch_size, in_channels, self.image_size, self.image_size)
        x = x.view(batch_size, -1, self.image_size, self.image_size)
        
        return x

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = 'cosine'
    ):
        """
        Args:
            denoise_fn (nn.Module): The denoiser (e.g., DiffusionTransformer).
            timesteps (int): Number of diffusion steps.
            beta_start (float): Starting beta value (only used for the linear schedule).
            beta_end (float): Ending beta value (only used for the linear schedule).
            schedule (str): Choice of noise schedule ('linear' or 'cosine').
        """
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.schedule = schedule
        
        if schedule == 'cosine':
            betas, alphas_cumprod = self.cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)            
        else:
            raise ValueError(f"Unknown schedule '{schedule}'. Use 'linear' or 'cosine'.")

        # shape (timesteps,)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    @staticmethod
    def cosine_beta_schedule(
        timesteps: int,
        s: float = 0.008
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the beta schedule using a cosine function following
            "Improved Denoising Diffusion Probabilistic Models" by Nichol and Dhariwal.
        Returns both the betas and the cumulative product of the alphas.
        
        Args:
            timesteps (int): Number of diffusion steps.
            s (float): Small offset parameter to adjust the schedule.
            
        Returns:
            betas (torch.Tensor): A tensor of betas of shape (timesteps,).
            alphas_cumprod (torch.Tensor): The cumulative product of (1 - betas), of shape (timesteps,).
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        # Compute cumulative alphas using a cosine schedule.
        alphas_cumprod_all = torch.cos(((x / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod_all = alphas_cumprod_all / alphas_cumprod_all[0]
        # Compute betas as the difference between successive cumulative products and clamp to prevent singularities near x = timesteps.
        betas = 1 - (alphas_cumprod_all[1:] / alphas_cumprod_all[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        # Use the computed alphas_cumprod (excluding the first element) as our schedule.
        alphas_cumprod = alphas_cumprod_all[1:]
        return betas, alphas_cumprod

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute x_t given x_0 and timestep t.
        
        Args:
            x0 (torch.Tensor): The original data.
            t (torch.Tensor): The timestep(s) to advance in the diffusion.
            noise (torch.Tensor): The noise to add to x0.
            
        Returns:
            xt (torch.Tensor): The noisy data.
            noise (torch.Tensor): The noise.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # find the appropriate weights
        # \sqrt{\overline{\alpha}_t}
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) # \sqrt{\overline{\alpha}_t}
        # \sqrt{1 - \overline{\alpha}_t}
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def p_losses(
        self,
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the MSE loss between the predicted noise and the actual noise.
        
        Args:
            x0 (torch.Tensor): The original data.
            t (torch.Tensor): The timestep(s) to advance in the diffusion, of shape (N,).
            noise (torch.Tensor): The noise to add to x0.
            
        Returns:
            loss (torch.Tensor)
        """
        xt, noise = self.forward_diffusion(x0, t, noise)
        predicted_noise = self.denoise_fn(xt, t.float())
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def reverse_diffusion(
        self,
        x_T: torch.Tensor,
        device: torch.device,
        sample_steps: Optional[list] = None,
        return_pil: bool = False,
        save_intermediate: Optional[int] = None
    ) -> Union[torch.Tensor, Image.Image, List[torch.Tensor], List[Image.Image]]:
        """
        Perform the reverse diffusion process to generate a sample from noisy input x_T.

        Args:
            x_T (torch.Tensor): The image to be denoised, of shape (C, W, H).
            device (torch.device): The device to perform computations on.
            sample_steps (Optional[list]): A list of timesteps to use for the reverse diffusion.
                If None, the full schedule (0, 1, ..., diffusion.timesteps-1) is used.
                No support for a sparse sampling schedule at the moment.
            return_pil (bool, Optional): If True, converts the denoised tensors to PIL images.
                Defaults to False.
            save_intermediate (Optional[int]): Periodically save the images in the reverse diffusion process.
                Saves an image once every save_intermediate timesteps. Always includes the initial image x_T
                and the last image x_0.
        Returns:
            Union[torch.Tensor, List[PIL.Image.Image]]:
                If `return_pil` is False, returns a torch.Tensor containing the reconstructed image
                    with shape (C, W, H).
                If `return_pil` is True, returns a PIL.Image.Image objects using the function
                    tensor_to_pil() in ../utils/image_processing.py 
        """
        self.eval()

        x = x_T.to(device)
        x = x.unsqueeze(0)
        # Use the full set of timesteps if no custom schedule is provided.
        if sample_steps is None:
            sample_steps = list(range(self.timesteps))
        # Ensure the steps are in ascending order (we'll iterate in reverse)
        sample_steps = sorted(sample_steps)
                
        if save_intermediate:
            assert len(sample_steps) % save_intermediate == 0, "Number of timesteps must be divisible by save_intermediate"
                    
        if save_intermediate:
            progression = [x]
            for t in tqdm(reversed(sample_steps), desc="Reverse diffusion", total=len(sample_steps)):
                t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
                with torch.no_grad():
                    predicted_noise = self.denoise_fn(x, t_tensor.float())
                
                beta_t = self.betas[t].item()
                alpha_t = 1 - beta_t
                alpha_bar_t = self.alphas_cumprod[t].item()
                
                # Compute the predicted mean (mu_theta) using the DDPM reverse formula:
                #   mu_theta = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1 - alpha_bar_t)) * predicted_noise)
                mu_theta = (1.0 / math.sqrt(alpha_t)) * (
                    x - (beta_t / math.sqrt(1 - alpha_bar_t)) * predicted_noise
                )
                
                sigma_t = math.sqrt(beta_t) if t > 0 else 0.0
                noise = torch.randn_like(x) if t > 0 else 0.0
                
                # Update x for the next timestep.
                x = mu_theta + sigma_t * noise
                if t % save_intermediate == 0:
                    progression.append(x)
            
            if t % save_intermediate != 0:
                    progression.append(x)
        
        else:
            for t in tqdm(reversed(sample_steps), desc="Reverse diffusion", total=len(sample_steps)):
                t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
                with torch.no_grad():
                    predicted_noise = self.denoise_fn(x, t_tensor.float())
                
                beta_t = self.betas[t].item()
                alpha_t = 1 - beta_t
                alpha_bar_t = self.alphas_cumprod[t].item()
                
                # Compute the predicted mean (mu_theta) using the DDPM reverse formula:
                #   mu_theta = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1 - alpha_bar_t)) * predicted_noise)
                mu_theta = (1.0 / math.sqrt(alpha_t)) * (
                    x - (beta_t / math.sqrt(1 - alpha_bar_t)) * predicted_noise
                )
                
                sigma_t = math.sqrt(beta_t) if t > 0 else 0.0
                noise = torch.randn_like(x) if t > 0 else 0.0
                
                # Update x for the next timestep.
                x = mu_theta + sigma_t * noise
                
        if return_pil:
            if save_intermediate:
                return batch_to_pil(torch.cat(progression, dim=0))
            else:
                return tensor_to_pil(x.squeeze(0))
        
        if save_intermediate:
            return [xt.squeeze(0) for xt in progression]
        else:
            return x.squeeze(0)
            
    def forward_diffusion_cascade(
        self,
        x0: torch.Tensor,
        device: torch.device,
        rate: int = 100,
        return_pil: bool = False,
    ) -> Union[List[torch.Tensor], List[Image.Image]]:
        """
        Compute the forward diffusion of a single image tensor step by step starting from x0.
        
        Args:
            x0 (torch.Tensor): The original image.
            rate (int): Saves a frame every {rate} timesteps.
            return_pil (bool, Optional): If True, converts the denoised tensors to PIL images.
                Defaults to False.
        Returns:
            progression (List[torch.Tensor]): A list containing the periodic frames from the forward diffusion.
        """
        
        x = x0.to(device)
        
        progression = [x]

        print(x.device)
        
        sqrt_betas = torch.sqrt(self.betas)
        one_minus_betas = 1 - self.betas
        sqrt_one_minus_betas = torch.sqrt(one_minus_betas)

        print(sqrt_betas.device, one_minus_betas.device, sqrt_one_minus_betas.device)
        
        for t in range(self.timesteps):
            noise = torch.randn_like(x)
            print(noise.device)
            x = sqrt_one_minus_betas[t]*x + sqrt_betas[t]*noise
            
            if t+1 % rate == 0 or t+1 == self.timesteps:
                progression.append(x)
        
        if return_pil:
            return batch_to_pil(torch.cat(progression, dim=0))
        
        return progression

def train_diffusion(
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    save_intermediate: Optional[int] = None,
    logdir: str = './weights',
    timestep_range: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Train the diffusion model.

    Args:
        diffusion (GaussianDiffusion): The diffusion wrapper.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Choice of optimizer (e.g., optimizer = torch.optim.Adam(diffusion.denoise_fn.parameters(), lr=1e-4))
        device (torch.device): Device to train on.
        epochs (int): Number of training epochs.
        timestep_range (Optional[Tuple[int, int]], optional): If provided, the model is only trained on timesteps in the range timestep_range.
            Otherwise, the training defaults to the entire range of timesteps [0, diffusion.timesteps].
            For use in finetuning on later times.
        save_intermediate (Optional[int], optional): If provided, save the denoising function's parameters every save_intermediate epochs.
            Otherwise, does not save any intermediate denoisers.
            Uses the same path as logdir.
        logdir (str, optional): Where to save the denoiser's parameters at the end of training.
            Defaults to a folder in the current directory called weights (the folder will be created if it does not exist).
    """
    diffusion.train()
    loss_record_by_epoch = []
    
    # If using a CUDA device, set up mixed precision training.
    scaler = torch.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Create dictionary to save model parameters
    checkpoint = {
        "diffusion_state_dict": diffusion.state_dict(),
        "hparams": {
            "timesteps": diffusion.timesteps,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "schedule": diffusion.schedule,
            # Hyperparameters for the denoiser network:
            "image_size": diffusion.denoise_fn.image_size,
            "patch_size": diffusion.denoise_fn.patch_size,
            "in_channels": diffusion.denoise_fn.in_channels,
            "emb_dim": diffusion.denoise_fn.emb_dim,
            "depth": diffusion.denoise_fn.depth,
            "nheads": diffusion.denoise_fn.nheads,
            "mlp_ratio": diffusion.denoise_fn.mlp_ratio,
            "time_emb_dim": diffusion.denoise_fn.time_emb_dim,
        }
    }    
    
    if timestep_range:
        assert (timestep_range[0] >= 0 and timestep_range[1] <= diffusion.timesteps), "Can only sample times in range(0, diffusion.timesteps)"
    else:
        timestep_range = (0, diffusion.timesteps)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Wrap the dataloader with tqdm for progress monitoring.
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device)
            # Sample random timesteps for each image in the batch.
            t = torch.randint(timestep_range[0], timestep_range[1], (batch.size(0),), device=device)
            
            optimizer.zero_grad()

            # Use mixed precision if available.
            if scaler is not None:
                with torch.autocast('cuda'):
                    loss = diffusion.p_losses(batch, t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = diffusion.p_losses(batch, t)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            # Uncomment to update progress bar with the latest loss.
            # progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        loss_record_by_epoch.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
        if save_intermediate and ((epoch+1) % save_intermediate == 0):
            logpath = os.path.join(logdir, f'diffusion_state_dict_{epoch+1}_of_{epochs}_' + time.strftime("%d-%m-%Y_%H-%M-%S") + ".pth")    
            checkpoint["diffusion_state_dict"] = diffusion.state_dict()
            torch.save(checkpoint, logpath)

    if save_intermediate and (epochs % save_intermediate):
        logpath = os.path.join(logdir, f'diffusion_state_dict_{epochs}_of_{epochs}_' + time.strftime("%d-%m-%Y_%H-%M-%S") + ".pth")
        checkpoint["diffusion_state_dict"] = diffusion.state_dict()
        torch.save(checkpoint, logpath)
    loss_array = np.array(loss_record_by_epoch)
    np.save(os.path.join(logdir, "loss_record" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".npy"), loss_array)

def late_train_diffusion(
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_weights: torch.Tensor,    
    epochs: int = 100,
    save_intermediate: Optional[int] = None,
    logdir: str = './weights'
) -> None:
    """
    Train the diffusion model.

    Args:
        diffusion (GaussianDiffusion): The diffusion wrapper.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Choice of optimizer (e.g., optimizer = torch.optim.Adam(diffusion.denoise_fn.parameters(), lr=1e-4))
        device (torch.device): Device to train on.
        epochs (int): Number of training epochs.
        time_weights (torch.Tensor): The weights for the timesteps in range(0, diffusion.timesteps), of shape (diffusion.timesteps,).
            Will be used to sample a multinomial distribution.
        save_intermediate (Optional[int], optional): If provided, save the denoising function's parameters every save_intermediate epochs.
            Otherwise, does not save any intermediate denoisers.
            Uses the same path as logdir.
        logdir (str, optional): Where to save the denoiser's parameters at the end of training.
            Defaults to a folder in the current directory called weights (the folder will be created if it does not exist).
    """
    diffusion.train()
    loss_record_by_epoch = []
    
    # If using a CUDA device, set up mixed precision training.
    scaler = torch.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Create dictionary to save model parameters
    checkpoint = {
        "diffusion_state_dict": diffusion.state_dict(),
        "hparams": {
            "timesteps": diffusion.timesteps,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "schedule": diffusion.schedule,
            # Hyperparameters for the denoiser network:
            "image_size": diffusion.denoise_fn.image_size,
            "patch_size": diffusion.denoise_fn.patch_size,
            "in_channels": diffusion.denoise_fn.in_channels,
            "emb_dim": diffusion.denoise_fn.emb_dim,
            "depth": diffusion.denoise_fn.depth,
            "nheads": diffusion.denoise_fn.nheads,
            "mlp_ratio": diffusion.denoise_fn.mlp_ratio,
            "time_emb_dim": diffusion.denoise_fn.time_emb_dim,
        }
    }    
    
    time_weights = time_weights.float().to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Wrap the dataloader with tqdm for progress monitoring.
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device)
            # Sample random timesteps for each image in the batch.
            t = torch.multinomial(time_weights, batch.size(0), replacement=True)
            
            optimizer.zero_grad()

            # Use mixed precision if available.
            if scaler is not None:
                with torch.autocast('cuda'):
                    loss = diffusion.p_losses(batch, t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = diffusion.p_losses(batch, t)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            # Uncomment to update progress bar with the latest loss.
            # progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        loss_record_by_epoch.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
        if save_intermediate and ((epoch+1) % save_intermediate == 0):
            logpath = os.path.join(logdir, f'diffusion_state_dict_{epoch+1}_of_{epochs}_' + time.strftime("%d-%m-%Y_%H-%M-%S") + ".pth")    
            checkpoint["diffusion_state_dict"] = diffusion.state_dict()
            torch.save(checkpoint, logpath)

    if save_intermediate and (epochs % save_intermediate):
        logpath = os.path.join(logdir, f'diffusion_state_dict_{epochs}_of_{epochs}_' + time.strftime("%d-%m-%Y_%H-%M-%S") + ".pth")
        checkpoint["diffusion_state_dict"] = diffusion.state_dict()
        torch.save(checkpoint, logpath)
    loss_array = np.array(loss_record_by_epoch)
    np.save(os.path.join(logdir, "loss_record" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".npy"), loss_array)

############################################
# Example Usage
############################################

if __name__ == "__main__":
    # Hyperparameters
    image_size = 64
    patch_size = 8
    in_channels = 3
    emb_dim = 256
    depth = 6
    nheads = 8
    timesteps = 1000
    batch_size = 32
    epochs = 1  # For demonstration; increase as needed
    # save_intermediate = 500
    
    # Import modules
    import torch
    from models import diffusion_transformer
    from utils import image_processing
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the transformer-based denoiser
    denoise_fn = diffusion_transformer.DiffusionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        emb_dim=emb_dim,
        depth=depth,
        nheads=nheads
    ).to(device)

    # Instantiate the Gaussian diffusion model
    diffusion = diffusion_transformer.GaussianDiffusion(denoise_fn, timesteps=timesteps).to(device)

    # Create a dummy dataset of random images (replace with your actual dataset)
    dummy_images = torch.randn(500, in_channels, image_size, image_size)  # 500 images
    dataloader = DataLoader(dummy_images, batch_size=batch_size, shuffle=True)

    # Optimizer (only update the denoiser parameters)
    optimizer = torch.optim.Adam(denoise_fn.parameters(), lr=1e-4)

    # Train the model
    train_diffusion(diffusion, dataloader, optimizer, device, epochs=epochs)

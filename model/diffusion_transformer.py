import torch
import torch.nn as nn

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
        self.num_patches = (image_size // patch_size) ** 2
        self.emb_dim = emb_dim

        # Convolution layer
        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))

        # Time embedding: sinusoidal embedding followed by MLP.
        self.time_emb_dim = time_emb_dim
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

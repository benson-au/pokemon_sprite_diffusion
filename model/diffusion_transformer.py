import torch

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

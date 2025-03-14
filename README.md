# T-DDPM
A transformer based implementation of the DDPM model from

> **Ho, Jonathan and Jain, Ajay and Abbeel, Pieter.**  
> *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems **33** (2020).  
> [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

In particular, we replace the U-Net in the denoising network in the reverse diffusion process with a transformer encoder. We also include options for the [cosine noise schedule](https://arxiv.org/pdf/2102.09672#equation.3.17) and the [sparse sampling schedule](https://arxiv.org/pdf/2102.09672#section.4) from

> **Nichol, Alexander Quinn and Dhariwal, Prafulla.**  
> *Improved Denoising Diffusion Probabilistic Models.* Proceedings of the 38th International Conference on Machine Learning, PMLR **139**:8162-8171, 2021.  
> [https://arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)

Our denoising architecture can be visualized as follows:

## Example: Gotta Diffuse 'Em All (Pok&eacute;mon Image Generation)

As a demonstration, we train our model on a dataset of Pok&eacute;mon images. Contrary to the [old motto](https://www.youtube.com/watch?v=R4GIyJxvk94) from the first season of the Pok&eacute;mon anime, we do not aim to be "the very best, like no one ever was". Instead, we train a relatively small model (X parameters) on a subset of images from [PokeAPI/sprites repository (official-artwork)](https://github.com/PokeAPI/sprites/tree/master/sprites/pokemon/other/official-artwork): we remove duplicates, Mega Pok&eacute;mon, Gigantamax Pok&eacute;mon, and make some other minor editorial choices (three pictures of Paldean Tauros?). The resulting dataset has 1293 - 158 = 1135 images. We scale the images, originally 475x475 pixels, to 256x256 pixels. We include the full list of parameter choices below. We train our model for N epochs. The resulting model weights are located at . Our experiment can be fully replicated using the following block of code:

### Model Parameters

#### DiffusionTransformer Parameters

| Parameter    | Value  | Description |
|--------------|--------|-------------|
| image_size   |     | Resolution of (square) input images. |
| patch_size   |      | Size of each image patch. |
| in_channels  |       | Number of channels. |
| emb_dim      |     | Embedding dimension for patch representations. |
| depth        |      | Number of transformer layers. |
| nheads       |      | Number of attention heads per layer. |
| mlp_ratio    | 4.0    | Ratio between the MLP hidden dimension and the embedding dimension. |

#### GaussianDiffusion Parameters

| Parameter  | Value         | Description |
|------------|---------------|-------------|
| denoise_fn | DiffusionTransformer instance | The denoising network used during diffusion. |
| timesteps  | 1000          | Number of diffusion steps. |
| schedule   | [cosine](https://arxiv.org/pdf/2102.09672#equation.3.17) | Noise schedule type. |

### DDPKM Samples

#### Samples Starting From Pure Noise

#### Samples Starting From Training Images

#### Samples Starting From Unseen Pok&eacute;mon Images

#### Samples Starting From Samples

## Data Sources

Training images used in this project were obtained from the [PokeAPI/sprites repository](https://github.com/PokeAPI/sprites).

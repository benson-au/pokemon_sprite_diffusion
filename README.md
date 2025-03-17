# Pok&eacute;mon sprite diffusion
(INSERT BEST IMAGE GENERATION HERE)

We implement a simple version of the diffusion model from

> **Ho, Jonathan and Jain, Ajay and Abbeel, Pieter.**  
> *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems **33** (2020).  
> [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

We replace the U-Net in the denoising network in the reverse diffusion process with a transformer. We also use the [cosine noise schedule](https://arxiv.org/pdf/2102.09672#equation.3.17) from

> **Nichol, Alexander Quinn and Dhariwal, Prafulla.**   
> *Improved Denoising Diffusion Probabilistic Models.* Proceedings of the 38th International Conference on Machine Learning, PMLR **139**:8162-8171, 2021.  
> [https://arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)

We train our model on a subset of Pok&eacute;mon sprites from the second generation of Pok&eacute;mon games: Gold, Silver, and Crystal. We use the front facing sprites sourced from [PokeAPI/sprites/sprites/pokemon/versions/generation-ii](https://github.com/PokeAPI/sprites/tree/master/sprites/pokemon/versions/generation-ii), removing all 26 of the [Unown](https://bulbapedia.bulbagarden.net/wiki/Unown_(Pok%C3%A9mon)) variants. The resulting dataset has 750 images with each unique Pok&eacute;mon appearing in two to three poses (most of the sprites in Pok&eacute;mon Crystal are either slight modifications or outright copies of the sprites from either Pok&eacute;mon Gold or Pok&eacute;mon Silver). As the sprites vary in resolution (40x40, 48x48, ), we scale them all to 64x64 pixels.

Contrary to the [motto](https://www.youtube.com/watch?v=R4GIyJxvk94) from the first season of the Pok&eacute;mon anime, we do not aim to be "the very best, like no one ever was". Instead, we train a relatively small model with N parameters for 5000 epochs. We include the full list of parameter choices below. The resulting model weights at intervals of 1000 epochs are located at .

## Model generations

Insert table of reverse diffusion samples (x-axis: timesteps, y-axis: model epochs) 

## Model parameters

#### DiffusionTransformer parameters

| Parameter    | Value  | Description |
|--------------|--------|-------------|
| image_size   |  64    | Resolution of (square) input images. |
| patch_size   |  4     | Size of each image patch. |
| in_channels  |  3     | Number of channels. |
| emb_dim      |  288   | Embedding dimension for patch representations. |
| depth        |  16    | Number of transformer layers. |
| nheads       |  8     | Number of attention heads per layer. |
| mlp_ratio    |  4     | Ratio between the MLP hidden dimension and the embedding dimension. |

#### GaussianDiffusion parameters

| Parameter  | Value         | Description |
|------------|---------------|-------------|
| denoise_fn | DiffusionTransformer instance | The denoising network used during diffusion. |
| timesteps  | 1000          | Number of diffusion steps. |
| schedule   | [cosine](https://arxiv.org/pdf/2102.09672#equation.3.17) | Noise schedule type. |

## Complication and remedy
The lack of diversity in our small dataset leads to stability issues during image generation. In particular, the model is highly sensitive to the standard Gaussian input `noise` (of shape `(batch_size, in_channels, W, H)`). We remedy this by normalizing `noise -= noise.mean(dim = 0)`. We observe substantial improvements in the empirical performance of the model with this minor modification.

(PICTURE OF FIXED VERSUS UNFIXED)

## Future directions
The model is limited by, among other things, computing resources and the paucity of samples. One could consider enlarging the dataset by including sprites from Generation I, but the style of the sprites is inconsistent with Generation II. Generations III-V are all of a similar style and can reasonably be lumped together to produce a much larger dataset (3000+ images).

## Data sources
Training images used in this project were obtained from the [PokeAPI/sprites repository](https://github.com/PokeAPI/sprites).

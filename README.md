# Pok&eacute;mon Sprite Diffusion
(INSERT BEST IMAGE GENERATION HERE)

We implement a simple version of the diffusion model from

> **Ho, Jonathan and Jain, Ajay and Abbeel, Pieter.**  
> *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems **33** (2020).  
> [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

We replace the U-Net in the denoising network in the reverse diffusion process with a transformer. We also use the [cosine noise schedule](https://arxiv.org/pdf/2102.09672#equation.3.17) from

> **Nichol, Alexander Quinn and Dhariwal, Prafulla.**   
> *Improved Denoising Diffusion Probabilistic Models.* Proceedings of the 38th International Conference on Machine Learning, PMLR **139**:8162-8171, 2021.  
> [https://arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)

We train our model on a subset of the Pok&eacute;mon sprites from the second generation of Pok&eacute;mon games: Gold, Silver, and Crystal. We use the front facing sprites sourced from [PokeAPI/sprites/sprites/pokemon/versions/generation-ii](https://github.com/PokeAPI/sprites/tree/master/sprites/pokemon/versions/generation-ii), removing all 26 of the [Unown](https://bulbapedia.bulbagarden.net/wiki/Unown_(Pok%C3%A9mon)) variants. The resulting dataset has 750 images with each unique Pok&eacute;mon appearing in two to three poses (most of the sprites in Pok&eacute;mon Crystal are either slight modifications or outright copies of the sprites from either Pok&eacute;mon Gold or Pok&eacute;mon Silver). As the sprites vary in resolution (40x40, 48x48, ), we scale them all to 64x64 pixels.

Contrary to the [motto](https://www.youtube.com/watch?v=R4GIyJxvk94) from the first season of the Pok&eacute;mon anime, we do not aim to be "the very best, like no one ever was". Instead, we train a relatively small model with N parameters for 5000 epochs. We include the full list of parameter choices below. The resulting model weights are located at .

## Model Generations

Insert table of reverse diffusion samples (x-axis: timesteps, y-axis: model epochs) 

## Model Parameters

#### DiffusionTransformer Parameters

| Parameter    | Value  | Description |
|--------------|--------|-------------|
| image_size   |  64    | Resolution of (square) input images. |
| patch_size   |  4     | Size of each image patch. |
| in_channels  |  3     | Number of channels. |
| emb_dim      |  288   | Embedding dimension for patch representations. |
| depth        |  16    | Number of transformer layers. |
| nheads       |  8     | Number of attention heads per layer. |
| mlp_ratio    |  4.0   | Ratio between the MLP hidden dimension and the embedding dimension. |

#### GaussianDiffusion Parameters

| Parameter  | Value         | Description |
|------------|---------------|-------------|
| denoise_fn | DiffusionTransformer instance | The denoising network used during diffusion. |
| timesteps  | 1000          | Number of diffusion steps. |
| schedule   | [cosine](https://arxiv.org/pdf/2102.09672#equation.3.17) | Noise schedule type. |

## Complications and suggestions for future directions
The model is limited by, among other things, computing resources and the paucity of samples. One could consider enlarging the dataset by including sprites from Generation I, but the style of the sprites is inconsistent with Generation II. Generations III-V are of similar style and can reasonably be lumped together to produce a much larger dataset (3000+ images).

The small dataset also leads to issues during image generation. In particular, after transforming the dataset using transforms.ToTensor() and linearly scaling the values to the range [-1, 1], the resulting tensor has nontrivial channel means [0.4799, 0.3649, 0.3370]. Thus, starting the reverse diffusion process from a zero mean Gaussian seems inappropriate. Indeed, this leads to poor performance, even on the fully trained model. We consider two remedies for this:

## Data Sources

Training images used in this project were obtained from the [PokeAPI/sprites repository](https://github.com/PokeAPI/sprites).

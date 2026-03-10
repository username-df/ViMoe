# ViMoe From Scratch

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/16d9122e-390e-4e1c-8614-8bfefb6e2d02" width="500" style="margin-right: 20px;" />
</div>

The transformer is usually used for natural language processing, but the Vision Transformer (ViT) is the application of the transformer encoder for computer vision tasks. 

## Patch Embedding

The first step of the Vision Transformer is turning the image into patches and embedding them.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/816302ac-2800-478c-bcb5-42cbb03e58e9" width="300" style="margin-right: 20px;" />
</div>

<br></br>

Using the einops library, images can be reshaped from (batch_size, channels, height, width) to (batch_size, num_patches, dimensionality of each patch = 3 * patch_size * patch_size)

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/cad1f48d-44e6-4c34-99a7-17273bac6638" width=700" style="margin-right: 20px;" />
</div>

<br></br>

The patches are then lineary embedded by using a nn.Linear layer which takes each patch and outputs a embedding vector.

(batch_size, num_patches, dimensionality of each patch) * (dimensionality of each patch, embed_vector_size) -> (batch_size, num_patches, embed_vector_size)

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/2734da53-a8ad-4afc-ae7d-81a6cd64e51e" width=700 style="margin-right: 20px;" />
</div>

<br></br>

Positional encoding is important for the transformer since it doesn't look at sequences step by step like RNNs, it looks at the whole sequence in parallel, so positional information needs to be explicitly added.
In the original Transformer paper, this was done manually using sine and cosine functions; the Vision Transformer uses a learnable positional encoding.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/fb728463-657b-4bd9-b8ab-74649570249c" width=850 style="margin-right: 20px;" />
</div>
<br></br>

The CLS token is also used, similar to BERT. The CLS token summarizes the information obtained from the whole sequence.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/f1063545-4959-4aeb-8b09-94246eee7938" width=800 style="margin-right: 20px;" />
</div>

<br></br>

The CLS token is concatenated to the start of the embedded patch sequence and the positional encoding is added to get the final patch embedding.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/8ac338f9-9d30-4295-b931-5acd06a0ba04" width=450 style="margin-right: 20px;" />
</div>

## Multi-Head Attention

The patch embedding is then used as input to the Transformer Encoder. The main component of the Transformer Encoder is Multi-Head Attention.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/9feced5f-a183-44fc-9d21-80ee5b7d8773" width=200 style="margin-right: 20px;" />
</div>

Self-Attention allows the Vision Transformer to focus on the relevant parts of the patch sequence by calculating a weighted sum of transformed patch embeddings. This prioritizes parts of the image containing important features like edges or textures, while reducing the impact of less important parts.

This done by creating queries (Q), keys (K), and values (V) from linear projections of the original sequence. A dot product between the queries and keys is calculated and then scaled by the square root of the dimensionality of the keys. A softmax is then used to normalize the scores between 0 and 1. A weighted sum between the softmax scores and the values is computed to see the importance of each patch; the CLS token summarizes the information from the weighted sum.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/a66dd65d-86b5-4b6a-b9ce-b6aee4173c64" width=400 style="margin-right: 20px;" />
</div>

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/ef3fa0a1-c333-4a66-92b2-d33372dd768c" width=400 style="margin-right: 20px;" />
</div>

Multi-Head Attention is an extension of Self-Attention, where instead of doing Self-Attention once, it is done h times.  The dimensionality of the model is split evenly across h heads; the dimensionality of each head is d_model / h. Each head performs self-attention independently. The output of all the heads are concatenated into a single matrix and passed through another linear layer to produce the final output.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/3f029e55-7092-4112-ad9e-2fb819c40a0e" width=650 style="margin-right: 20px;" />
</div>

## Mixture of Experts

<img width="520" height="520" alt="b6a623a4-fdbc-4abf-883b-3c2679b4ad4d_1460x640" src="https://github.com/user-attachments/assets/ac2c37e7-1e42-4ed7-b067-00a457b61190" />

Mixture of Experts (MOE) works by utilizing a router to assign a small number of experts to each patch in the image, the output for that patch is a weighted sum of the outputs from the selected experts. The idea is to only activate specific experts when certain shapes or textures are found in images.

To prevent expert collapse, where only a few experts are used for every single patch, auxiliary-loss free load balancing is used to ensure the number of patches assigned to each expert is around the expected average. This is done by using a bias term to increase the probability of selecting the experts that were not used as much.

<img width="682" height="841" alt="Screenshot 2026-03-10 115814" src="https://github.com/user-attachments/assets/ae2095aa-84bf-4999-ad37-91087c75962a" />

## Transformer Encoder

The Transformer Encoder takes Multi-Head Attention and adds skip connections, layer norm and a multi-layer perceptron (MLP). 

<img width="1208" height="200" alt="image" src="https://github.com/user-attachments/assets/6691b18d-847b-4721-8a1c-8e51382d4ed9" />

In every other encoder block, Sparse Mixture of Experts replaces the MLP; this allows for lower computational costs while maintaining overall model performance and quality, which improves model scalability.

## Vision Transformer (ViT)

The Vision Transformer utilizes multiple encoder blocks and uses the final CLS token as input to a MLP, which acts as a classifier for the image.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/e46abc9b-db61-419c-a6f2-44054033a75b" width=550 style="margin-right: 20px;" />
</div>

## Training

To train a Vision Transformer, multiple regularization techniques are required. The following were used: 

- data augmentation (horizontal flip, color jitter, cropping, normalization)
- drop out after linear layers
- weight decay/L2 regularization ( + AdamW optimizer instead of Adam)
- cosine learning rate scheduler
- label smoothing

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/750ac2cd-ce0f-42ab-8336-3a130fc93704" width=550 style="margin-right: 20px;" />
</div>

The model achieved around 77% test accuracy on CIFAR10 after 100 epochs.

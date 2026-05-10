"""
IntentFormer model for k=3 nuScenes pedestrian intent.

Adapted from /root/IntentFormer/main.ipynb cells 21-25.
Key deltas vs. notebook:
  - INPUT_SHAPE = (3, 224, 224, 3)   (was (14, 224, 224, 3))
  - PATCH_SIZE  = (1, 8, 8)          (was (2, 8, 8) -- collapsed to 1 token at T=3)
  - Token count: T*28*28 = 3*784 = 2352 per modality (was 5488)

3 inputs: RGB sequence, segmentation sequence, bbox sequence (+ label passthrough).
3 outputs: rgb_o, seg_o, traj_o (each 2-class softmax).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42

INPUT_SHAPE = (3, 224, 224, 3)
INPUT_SHAPE2 = (3, 4)         # bbox sequence: (k, [cx,cy,w,h])
INPUT_SHAPE3 = (3, 1)         # label-pass / scalar slot (unused by graph)

PATCH_SIZE = (1, 8, 8)        # tubelet 3D conv stride; preserves temporal dim at k=3
PROJECTION_DIM = 64
NUM_HEADS = 4
LAYER_NORM_EPS = 1e-6
NUM_CLASSES = 2


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='VALID',
            kernel_initializer=keras.initializers.HeNormal(seed=123),
            activity_regularizer=keras.regularizers.L1(l1=1e-5),
        )
        self.projection2 = layers.GRU(embed_dim, return_sequences=True, return_state=True)
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos, vid):
        if vid == 0:
            projected_patches = self.projection(videos)
        elif vid == 1:
            projected_patches, _ = self.projection2(videos)
        else:
            raise ValueError(f'vid must be 0 or 1, got {vid}')
        return self.flatten(projected_patches)

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim,
                       'patch_size': self.patch_size})
        return config


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        return encoded_tokens + self.position_embedding(self.positions)

    def get_config(self):
        c = super().get_config()
        c.update({'embed_dim': self.embed_dim})
        return c


class PositionalEncoder2(PositionalEncoder):
    """Identical to PositionalEncoder; kept as separate class to match notebook
    layer-naming so future weight-loads from the published checkpoints align.
    """
    pass


class CustomCELoss(keras.layers.Layer):
    """Co-learning Composite (CAC) loss: weighted SCE across the 3 heads
    with learnable scalar weights w1, w2, w3.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w1 = tf.Variable(0.2, name='w1', trainable=True)
        self.w2 = tf.Variable(0.3, name='w2', trainable=True)
        self.w3 = tf.Variable(0.5, name='w3', trainable=True)

    def call(self, y_true, y_pred1, y_pred2, y_pred3):
        sce = keras.losses.SparseCategoricalCrossentropy()
        loss_r = sce(y_true, y_pred1)
        loss_rs = sce(y_true, y_pred2)
        loss_rst = sce(y_true, y_pred3)
        return (tf.cast(self.w1, tf.float32) * loss_r
                + tf.cast(self.w2, tf.float32) * loss_rs
                + tf.cast(self.w3, tf.float32) * loss_rst)


def _co_learn_block(query_encoded, kv_encoded, attn_layer, mlp,
                    num_heads, embed_dim):
    """One co-learning stage: cross-attend query <-> kv, then MLP residual.
    Returns (encoded_patches_after_block, classifier_logits).
    """
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(query_encoded)
    xn = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(kv_encoded)

    a1 = attn_layer(x1, xn)        # query attends to kv -> (B, T_q, D)
    a2 = attn_layer(xn, x1)        # kv   attends to query -> (B, T_kv, D)

    # PCP: project a2 from T_kv tokens up to T_q tokens via 1x1 conv on transposed dim.
    a2_t = layers.Permute((2, 1))(a2)                    # (B, D, T_kv)
    a2_t = layers.Conv1D(a1.shape[1], 1)(a2_t)            # (B, D, T_q)
    a2_p = layers.Permute((2, 1))(a2_t)                  # (B, T_q, D)
    attn_out = layers.Add()([a1, a2_p])                  # fused (B, T_q, D)

    x2 = layers.Add()([attn_out, query_encoded])
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
    x3 = mlp(x3)
    encoded = layers.Add()([x3, x2])

    rep = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded)
    rep = layers.GlobalAvgPool1D()(rep)
    rep = layers.Dropout(0.5)(rep)
    return encoded, rep


def build_intentformer(input_shape=INPUT_SHAPE,
                       input_shape2=INPUT_SHAPE2,
                       input_shape3=INPUT_SHAPE3,
                       patch_size=PATCH_SIZE,
                       embed_dim=PROJECTION_DIM,
                       num_heads=NUM_HEADS,
                       num_classes=NUM_CLASSES):
    """Build the IntentFormer model.

    Returns: keras.Model with:
        inputs  = [RGB(B,T,224,224,3), mask(B,T,224,224,3), box(B,T,4), label(B,T,1)]
        outputs = [rgb_o(B,2), seg_o(B,2), traj_o(B,2)]
    """
    tubelet_embedder = TubeletEmbedding(embed_dim=embed_dim, patch_size=patch_size)
    pos_enc = PositionalEncoder(embed_dim=embed_dim)
    pos_enc2 = PositionalEncoder2(embed_dim=embed_dim)

    shared_mlp = keras.Sequential([
        layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu,
                     kernel_initializer=keras.initializers.HeNormal(seed=SEED),
                     kernel_regularizer=tf.keras.regularizers.L2(1e-6)),
        layers.Dropout(0.5),
        layers.Dense(units=embed_dim, activation=tf.nn.gelu,
                     kernel_initializer=keras.initializers.HeNormal(seed=SEED),
                     kernel_regularizer=tf.keras.regularizers.L2(1e-6)),
    ])
    shared_attn1 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.5)
    shared_attn2 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.5)

    input_0 = layers.Input(shape=input_shape, name='RGB')
    input_1 = layers.Input(shape=input_shape, name='mask')
    input_2 = layers.Input(shape=input_shape2, name='box')
    label = layers.Input(shape=input_shape3, name='target')

    patches_0 = tubelet_embedder(input_0, 0)
    patches_1 = tubelet_embedder(input_1, 0)
    patches_2 = tubelet_embedder(input_2, 1)

    enc_0 = pos_enc(patches_0)
    enc_1 = pos_enc(patches_1)
    enc_2 = pos_enc2(patches_2)

    # Stage I: RGB self-attention
    encoded = enc_0
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded)
    self_attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.5)(x1, x1)
    x2 = layers.Add()([self_attn, encoded])
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
    x3 = shared_mlp(x3)
    encoded = layers.Add(name='encoded_R')([x3, x2])
    rep = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded)
    rep = layers.GlobalAvgPool1D()(rep)
    rep = layers.Dropout(0.5)(rep)
    output_r = layers.Dense(
        units=num_classes, activation='softmax', name='rgb_o',
        kernel_initializer=keras.initializers.HeNormal(seed=SEED))(rep)

    # Stage II: RGB <-> seg co-learning
    encoded, rep = _co_learn_block(encoded, enc_1, shared_attn1, shared_mlp,
                                   num_heads, embed_dim)
    output_s = layers.Dense(
        units=num_classes, activation='softmax', name='seg_o',
        kernel_initializer=keras.initializers.HeNormal(seed=SEED))(rep)

    # Stage III: (RGB+seg) <-> bbox co-learning
    encoded, rep = _co_learn_block(encoded, enc_2, shared_attn2, shared_mlp,
                                   num_heads, embed_dim)
    output_t = layers.Dense(
        units=num_classes, activation='softmax', name='traj_o',
        kernel_initializer=keras.initializers.HeNormal(seed=SEED))(rep)

    return keras.Model(inputs=[input_0, input_1, input_2, label],
                       outputs=[output_r, output_s, output_t],
                       name='IntentFormer_k3')


if __name__ == '__main__':
    m = build_intentformer()
    m.summary()
    n_tokens = ((INPUT_SHAPE[0] // PATCH_SIZE[0])
                * (INPUT_SHAPE[1] // PATCH_SIZE[1])
                * (INPUT_SHAPE[2] // PATCH_SIZE[2]))
    print(f'\ninput_shape={INPUT_SHAPE}  patch_size={PATCH_SIZE}')
    print(f'tokens per modality (RGB/seg) = {n_tokens}')

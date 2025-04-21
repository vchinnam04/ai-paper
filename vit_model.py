import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class CustomViT(tf.keras.Model):
    def __init__(self, image_size=224, patch_size=16, num_classes=101,
                 projection_dim=768, transformer_layers=12, num_heads=12,
                 mlp_dim=3072, dropout_rate=0.1):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_proj = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid'
        )
        self.flatten_patches = layers.Reshape((self.num_patches, projection_dim))
        self.encoder = PatchEncoder(self.num_patches, projection_dim)
        self.cls_token = self.add_weight("cls_token", shape=[1, 1, projection_dim], initializer="zeros")

        self.transformer_blocks = []
        for _ in range(transformer_layers):
            self.transformer_blocks.append([
                layers.LayerNormalization(epsilon=1e-6),
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate),
                layers.Dropout(dropout_rate),
                layers.LayerNormalization(epsilon=1e-6),
                models.Sequential([
                    layers.Dense(mlp_dim, activation=tf.nn.gelu),
                    layers.Dropout(dropout_rate),
                    layers.Dense(projection_dim),
                    layers.Dropout(dropout_rate)
                ])
            ])

        self.head = models.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(num_classes)
        ])

    def call(self, inputs, training=False):
        x = self.patch_proj(inputs)
        x = self.flatten_patches(x)

        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, tf.shape(x)[-1]])
        x = tf.concat([cls_tokens, x], axis=1)
        x = self.encoder(x)

        for norm1, attn, drop1, norm2, mlp in self.transformer_blocks:
            x1 = norm1(x)
            attn_out = attn(x1, x1)
            x = x + drop1(attn_out, training=training)

            x2 = norm2(x)
            x = x + mlp(x2, training=training)

        return self.head(x[:, 0])

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * step / tf.cast(self.warmup_steps, tf.float32)
        cosine_steps = tf.maximum(step - self.warmup_steps, 0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * cosine_steps / (self.total_steps - self.warmup_steps)))
        return tf.where(step < self.warmup_steps, warmup_lr, self.base_lr * cosine_decay)

def compile_vit(model, total_steps=10000, warmup_steps=500):
    lr_schedule = WarmupCosineSchedule(base_lr=1e-3, total_steps=total_steps, warmup_steps=warmup_steps)
    optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
        ]
    )

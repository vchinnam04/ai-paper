import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        cosine_steps = tf.maximum(step - self.warmup_steps, 0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * cosine_steps / (self.total_steps - self.warmup_steps)))
        decayed_lr = self.initial_lr * cosine_decay
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }

class FoodCNN:
    def __init__(self, num_classes, input_shape=(224, 224, 3), model_type="efficientnetv2-b0"):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_type = model_type.lower()
        self.model = self._build_model()

    def _get_backbone(self):
        if self.model_type == "efficientnetv2-b1":
            return EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_type == "efficientnetv2-b2":
            return EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        return EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=self.input_shape)

    def _build_model(self):
        base_model = self._get_backbone()
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes)(x)

        model = tf.keras.Model(inputs, outputs)
        model.summary()
        return model

    def compile_model(self):
        initial_learning_rate = 0.002
        warmup_epochs = 5
        total_steps = 10000

        lr_schedule = WarmupCosineDecay(initial_learning_rate, warmup_epochs * 100, total_steps)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
            nesterov=True
        )

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

    def _create_callbacks(self):
        return [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
            ModelCheckpoint(filepath=f'models/cnn_{self.model_type}_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
        ]

    def train(self, train_ds, val_ds, epochs=20, batch_size=64):
        callbacks = self._create_callbacks()

        print("\nTraining top layers...")
        base_model = self.model.layers[1]
        base_model.trainable = False
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks, batch_size=batch_size)

        print("\nFine-tuning all layers...")
        base_model.trainable = True
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=0.0001, momentum=0.9, nesterov=True),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
        )

        history_fine = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs - 5, callbacks=callbacks, batch_size=batch_size)

        for key in history_fine.history:
            history.history[key].extend(history_fine.history[key])
        return history

    def evaluate(self, val_ds):
        results = self.model.evaluate(val_ds)
        return {'loss': results[0], 'accuracy': results[1], 'top5_accuracy': results[2] if len(results) > 2 else None}

    def predict(self, image):
        image = tf.image.resize(image, self.input_shape[:2])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        image = tf.expand_dims(image, 0)
        return self.model.predict(image)

    def save_model(self, custom_path=None):
        os.makedirs('models', exist_ok=True)
        path = custom_path or f'models/cnn_{self.model_type}_model.keras'
        self.model.save_weights(path.replace('.keras', '.weights.h5'))
        self.model.save(path, save_format='keras')
        with open(path.replace('.keras', '_architecture.json'), 'w') as f:
            f.write(self.model.to_json())
        print(f"Model saved to {path}")

#!/usr/bin/env python3
import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import preprocess_data
from cnn_model import FoodCNN
from cnn_model_custom import CustomCNN
from vit_model import CustomViT, compile_vit

def parse_args():
    parser = argparse.ArgumentParser(description="Train Food Recognition Model")
    parser.add_argument("--mode", type=str, choices=["train"], default="train")
    parser.add_argument("--model", type=str, choices=["cnn", "vit", "custom", "customvit"], default="cnn")
    parser.add_argument("--model_type", type=str, choices=["efficientnetv2-b0", "efficientnetv2-b1", "efficientnetv2-b2"], default="efficientnetv2-b0")
    parser.add_argument("--data_limit", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def plot_training_history(history, output_path="training_plot.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training plot saved to {output_path}")
    plt.close()

def test_training(model_name, model_type, data_limit=0.2, epochs=20, batch_size=64):
    print("\nStep 1: Data Preprocessing")
    train_ds, val_ds, num_classes, class_weights, class_names = preprocess_data(
        data_limit=data_limit,
        img_size=224,
        apply_advanced_aug=True,
        batch_size=batch_size
    )

    print(f"\nStep 2: Training Model ({model_name.upper()})")

    if model_name == "cnn":
        model = FoodCNN(num_classes=num_classes, input_shape=(224, 224, 3), model_type=model_type)
        model.compile_model()
        history = model.train(train_ds, val_ds, epochs=epochs, batch_size=batch_size)
        model.save_model()

    elif model_name == "vit":
        model = FoodViT(num_classes=num_classes, input_shape=(224, 224, 3))
        model.compile_model(learning_rate=0.0005)
        history = model.train(train_ds, val_ds, epochs=epochs)
        model.save_model()

    elif model_name == "custom":
        model = CustomCNN(num_classes=num_classes, input_shape=(224, 224, 3))
        model.compile_model()
        history = model.train(train_ds, val_ds, epochs=epochs, batch_size=batch_size)
        model.save_model(path="models/custom_cnn_final.keras")

    elif model_name == "customvit":
        total_steps = epochs * len(train_ds)
        model = CustomViT(num_classes=num_classes)
        compile_vit(model, total_steps=total_steps, warmup_steps=500)
        train_batches = train_ds.prefetch(tf.data.AUTOTUNE)
        val_batches = val_ds.prefetch(tf.data.AUTOTUNE)
        history = model.fit(train_batches, validation_data=val_batches, epochs=epochs)
        model.save_weights("models/custom_vit_final.weights.h5")

    else:
        raise ValueError("Unsupported model type")

    plot_training_history(history)

def main():
    args = parse_args()
    print("FOOD RECOGNITION PROJECT")
    print(f"Model: {args.model.upper()} | Type: {args.model_type if args.model == 'cnn' else 'N/A'} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")

    if args.mode == "train":
        test_training(
            model_name=args.model,
            model_type=args.model_type,
            data_limit=args.data_limit,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()

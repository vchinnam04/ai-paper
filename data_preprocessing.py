import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

IMG_SIZE = 224
NUM_CLASSES = 101

def apply_normalization(image):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std

def advanced_augment(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = apply_normalization(image)
    return image, label

def simple_preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = apply_normalization(image)
    return image, label

def calculate_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes)
    for _, label in dataset:
        counts[label.numpy()] += 1
    total = np.sum(counts)
    weights = total / (num_classes * counts)
    return weights / np.mean(weights)

def plot_class_distribution(dataset, class_names, output_file="class_distribution.png"):
    counts = np.zeros(len(class_names))
    for _, label in dataset:
        counts[label.numpy()] += 1
    plt.figure(figsize=(18, 6))
    plt.bar(range(len(class_names)), counts)
    plt.title("Class Distribution (Limited Data Sample)")
    plt.xlabel("Class Index")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    plt.close()
    print(f"Class distribution plot saved to {output_file}")

def preprocess_data(data_limit=1.0, img_size=224, apply_advanced_aug=True, batch_size=32):
    print("\nLoading Food101 dataset...")
    (train_ds, val_ds), info = tfds.load(
        'food101',
        split=['train', 'validation'],
        as_supervised=True,
        with_info=True,
        shuffle_files=True
    )

    if data_limit < 1.0:
        train_ds = train_ds.take(int(info.splits['train'].num_examples * data_limit))
        val_ds = val_ds.take(int(info.splits['validation'].num_examples * data_limit))
        print(f"Using {data_limit*100:.1f}% of dataset")

    class_names = info.features['label'].names

    plot_class_distribution(train_ds, class_names)

    train_ds = train_ds.map(advanced_augment if apply_advanced_aug else simple_preprocess,
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(simple_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    class_weights = calculate_class_weights(train_ds.unbatch(), info.features['label'].num_classes)
    print("Class weights calculated.")

    return train_ds, val_ds, info.features['label'].num_classes, class_weights, class_names

def preprocess_single_image(image_path, img_size=224):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = apply_normalization(image)
    return tf.expand_dims(image, 0)

if __name__ == "__main__":
    train_ds, val_ds, num_classes, class_weights, class_names = preprocess_data(data_limit=0.05)
    print("Preprocessing test completed.")

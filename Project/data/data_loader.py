import tensorflow as tf
import os
from config.config import celeba_dir, BATCH_SIZE, BUFFER_SIZE

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [64, 64])
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def get_dataset():
    image_files = [os.path.join(celeba_dir, f) for f in os.listdir(celeba_dir) if f.endswith('.jpg')]
    train_dataset = tf.data.Dataset.from_tensor_slices(image_files)
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset
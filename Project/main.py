import tensorflow as tf
from data.data_loader import get_dataset
from models.generator import generator_model
from models.discriminator import discriminator_model
from training.train import train
from config.config import noise_dim, num_examples_to_generate, EPOCHS

# Initialize models
generator = generator_model()
discriminator = discriminator_model()

# Get dataset
train_dataset = get_dataset()

# Generate seed for image generation
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Start training
train(train_dataset, EPOCHS, generator, discriminator, seed)
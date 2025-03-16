import os
BATCH_SIZE = 128
BUFFER_SIZE = 60000
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
celeba_dir = r"C:\Users\moham\Downloads\archive\img_align_celeba\img_align_celeba"
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
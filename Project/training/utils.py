import matplotlib.pyplot as plt
from config.config import num_examples_to_generate

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = predictions * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1]
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    if epoch == EPOCHS:
        plt.savefig('final_image.png')
        plt.show()
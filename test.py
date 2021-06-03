# load model
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def plot_images(images):
    
    plt.figure(figsize=(10, 10))

    for i in range(images.shape[0]):
        plt.subplot(5, 5, i+1)
        image = images[i, :, :, :]
        image = (image + 1) / 2.0  # Rescale to [0, 255]
        plt.imshow(image)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

G = load_model('gan.h5')

new_images = G.predict(np.random.normal(0, 1, 16, 100))
plot_images(new_images)
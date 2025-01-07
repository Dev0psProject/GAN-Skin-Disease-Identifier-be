import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Define the generator model with 64x64 output shape
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128 * 16 * 16, activation="relu", input_shape=(100,)),
        layers.Reshape((16, 16, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(3, (3, 3), padding="same", activation="tanh")
    ])
    return model

# Define the discriminator model with 64x64 input shape
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=(64, 64, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = tf.keras.models.Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")
    return gan

# Load real images
def load_real_images(data_dir, batch_size):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    dataset = datagen.flow_from_directory(data_dir, target_size=(64, 64), batch_size=batch_size, class_mode=None)
    return dataset

# GAN training loop
def train_gan(generator, discriminator, gan, dataset, epochs=10000, batch_size=32):
    for epoch in range(epochs):
        # Load a batch of real images
        real_images = next(dataset)  
        
        # Continue if the batch size is inconsistent
        if real_images.shape[0] != batch_size:
            continue
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        
        # Ensure shapes of real and fake images match
        assert real_images.shape == fake_images.shape, \
            f"Real images shape: {real_images.shape}, Fake images shape: {fake_images.shape}"
        
        # Combine real and fake images
        images = np.concatenate([real_images, fake_images])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        
        # Train discriminator
        discriminator.trainable = True
        d_loss, d_acc = discriminator.train_on_batch(images, labels)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        misleading_labels = np.ones((batch_size, 1))  # Generator tries to 'fool' the discriminator
        g_loss = gan.train_on_batch(noise, misleading_labels)
        
        # Log progress
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss}, D Acc: {d_acc}, G Loss: {g_loss}")
            save_images(generator, epoch)

# Save generated images for inspection
def save_images(generator, epoch, num_images=10):
    noise = np.random.normal(0, 1, (num_images, 100))
    generated_images = generator.predict(noise)
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
    
    # Make sure the directory exists
    gan_images_path = "../dataset/gan_images"
    os.makedirs(gan_images_path, exist_ok=True)  # Ensure the directory exists
    for i, image_array in enumerate(generated_images):
        image = Image.fromarray(image_array)
        image.save(f"{gan_images_path}/generated_image_{epoch}_{i}.png")  # Save in the specified directory

# Directory with real skin disease images
data_dir = '../dataset/train'  # Update with the path to your training dataset
batch_size = 32

# Instantiate and compile models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Load dataset and train GAN
dataset = load_real_images(data_dir, batch_size)
train_gan(generator, discriminator, gan, dataset)

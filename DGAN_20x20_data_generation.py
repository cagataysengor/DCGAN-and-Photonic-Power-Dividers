#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:26:56 2023

@author: sengor
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display


train_images = np.load("selected_data_symmetric.npy")
ayiklanmis_images = np.load("ayiklanmis_fake_images_data_asymmetric.npy")

train_images = np.concatenate((train_images,ayiklanmis_images, axis=0)

train_images = train_images.reshape(train_images.shape[0], 20, 20, 1)

train_images = train_images[np.random.permutation(train_images.shape[0])]


BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 32

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# seed = 42
# tf.random.set_seed(seed)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*5*256, use_bias=True, input_shape=(300,), kernel_initializer=tf.keras.initializers.HeNormal())) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((5, 5, 256)))
    assert model.output_shape == (None, 5, 5, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal()))
    assert model.output_shape == (None, 5, 5, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal()))
    assert model.output_shape == (None, 10, 10, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform()))
    assert model.output_shape == (None, 20, 20, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 300])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[20, 20, 1], kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform()))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
noise_dim = 300
num_examples_to_generate = 1000

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Lists to store generator and discriminator losses for each epoch
gen_losses_history = []
disc_losses_history = []
    
prediction_images = []

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    gen_loss_sum = 0
    disc_loss_sum = 0
    
    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_loss_sum += gen_loss
      disc_loss_sum += disc_loss
      
    avg_gen_loss = gen_loss_sum / len(dataset)
    avg_disc_loss = disc_loss_sum / len(dataset)
    
    # Store losses for plotting
    gen_losses_history.append(avg_gen_loss.numpy())
    disc_losses_history.append(avg_disc_loss.numpy())
   
    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Generator Loss: {}, Discriminator Loss: {}'.format(gen_loss.numpy(), disc_loss.numpy()))
  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()
  np.save("prediction_images.npy",predictions)

train(train_dataset, EPOCHS)

# Plot generator and discriminator losses over epochs
plt.plot(range(1, EPOCHS + 1), gen_losses_history, label='Generator Loss')
plt.plot(range(1, EPOCHS + 1), disc_losses_history, label='Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

## Display a single image using the epoch number
#def display_image(epoch_no):
  #return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

#display_image(EPOCHS)



prediction_images = np.load("prediction_images.npy")
prediction_images = prediction_images.reshape(1000,20,20)
v_prediction_images = np.vectorize(lambda x: 1 if x > 0.5 else 0)(prediction_images)  
plt.imshow(np.rot90(v_prediction_images[0], k=1) ,cmap="gray")
np.save("v_prediction_images.npy",v_prediction_images)

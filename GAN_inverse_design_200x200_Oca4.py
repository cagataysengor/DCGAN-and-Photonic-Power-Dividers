#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:14:45 2024

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


train_images = np.load("upscaled_images.npy")

ayiklanmis_images42_200x200 = np.load("ayiklanmis_fake_images_data42_200x200_asymmetric.npy")
ayiklanmis_images43_200x200 = np.load("ayiklanmis_fake_images_data43_200x200_asymmetric.npy")
ayiklanmis_images44_200x200 = np.load("ayiklanmis_fake_images_data44_200x200_asymmetric.npy")
ayiklanmis_images45_200x200 = np.load("ayiklanmis_fake_images_data45_200x200_asymmetric.npy")
ayiklanmis_images46_200x200 = np.load("ayiklanmis_fake_images_data46_200x200_asymmetric.npy")


train_images = np.concatenate((train_images, ayiklanmis_images42_200x200,
                               
                               ayiklanmis_images45_200x200,ayiklanmis_images46_200x200
                               ))

train_images = train_images.reshape(train_images.shape[0], 200, 200, 1)

train_images = train_images[np.random.permutation(train_images.shape[0])]


# Assuming train_images is a NumPy array
# num_shuffles = 15  # You can adjust this number as needed

# for _ in range(num_shuffles):
#     np.random.shuffle(train_images)

# np.save(file_path_fake_images,train_images)

validation_images = train_images[60:]
train_images = train_images[:60]

# reshaped_data = train_images.reshape(-1, train_images.shape[2])

# # Count the number of columns that are the same
# num_same_columns = reshaped_data.shape[1] - np.unique(reshaped_data, axis=1).shape[1]

# print(f"Number of columns that are the same: {num_same_columns}")

BUFFER_SIZE = 60
BATCH_SIZE1 = 3
BATCH_SIZE2 = 3

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE1)
validation_dataset = tf.data.Dataset.from_tensor_slices(validation_images).batch(BATCH_SIZE2)
# seed = 42
# tf.random.set_seed(seed)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(50*50*256, use_bias=True, input_shape=(60,), kernel_initializer=tf.keras.initializers.HeNormal())) # , kernel_initializer=tf.keras.initializers.HeUniform()
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((50, 50, 256)))
    assert model.output_shape == (None, 50, 50, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal()))
    assert model.output_shape == (None, 50, 50, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal()))
    assert model.output_shape == (None, 100, 100, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform()))
    assert model.output_shape == (None, 200, 200, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 60])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[200, 200, 1], kernel_initializer=tf.keras.initializers.HeNormal()))
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
# cross_entropy = tf.keras.losses.KLDivergence()

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
noise_dim = 60
num_examples_to_generate = 100

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Variables for early stopping
early_stopping_counter = 0
early_stopping_patience = 10  # Number of epochs to wait for improvement # 10 yapılacak
best_gen_loss = float('inf')  # Initialize with a large value
    
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE1, noise_dim])

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

def validate(model, dataset):
    total_loss = 0
    for image_batch in dataset:
        noise = tf.random.normal([image_batch.shape[0], noise_dim])
        generated_images = model(noise, training=False)
        fake_output = discriminator(generated_images, training=False)
        total_loss += generator_loss(fake_output).numpy()
    return total_loss / len(dataset)
# Lists to store generator and discriminator losses for each epoch
gen_losses_history = []
disc_losses_history = []
val_gen_losses_history = []  
val_disc_losses_history = []
    
prediction_images = []

def train(train_dataset, validation_dataset, epochs):
  
  global early_stopping_counter, best_gen_loss
  
  for epoch in range(epochs):
    start = time.time()
    gen_loss_sum = 0
    disc_loss_sum = 0
    
    for image_batch in train_dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_loss_sum += gen_loss
      disc_loss_sum += disc_loss
      
    avg_gen_loss = gen_loss_sum / len(train_dataset)
    avg_disc_loss = disc_loss_sum / len(train_dataset)
    
    # Store losses for plotting
    gen_losses_history.append(avg_gen_loss.numpy())
    disc_losses_history.append(avg_disc_loss.numpy())

    # Validation
    val_gen_loss_sum = 0
    val_disc_loss_sum = 0
    
    for val_image_batch in validation_dataset:
            val_gen_loss, val_disc_loss = train_step(val_image_batch)
            val_gen_loss_sum += val_gen_loss
            val_disc_loss_sum += val_disc_loss
            
    avg_val_gen_loss = val_gen_loss_sum / len(validation_dataset)
    avg_val_disc_loss = val_disc_loss_sum / len(validation_dataset)    

    # Store validation losses for plotting
    val_gen_losses_history.append(avg_val_gen_loss.numpy())
    val_disc_losses_history.append(avg_val_disc_loss.numpy())    
    
   
   
    
   # Early stopping
    if epoch % 2 == 0:  # Validate every 2 epochs
        val_loss = validate(generator, validation_dataset)

        if val_loss < best_gen_loss:
            best_gen_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        print('Epoch {}: Generator Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, avg_gen_loss, val_loss))

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
            break
    
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
  # predictions = np.round(predictions)
  
  

  # fig = plt.figure(figsize=(4, 4))


  # for i in range(predictions.shape[0]):
  #      plt.subplot(4, 4, i+1)
  #      plt.imshow(predictions[i, :, :, 0] , cmap='gray')
  #      plt.axis('off')
      

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  np.save("prediction_images.npy",predictions)

train(train_dataset, validation_dataset, EPOCHS)

# Plot generator and discriminator losses over epochs
plt.plot(range(1, len(gen_losses_history) + 1), gen_losses_history, label='Generator Loss (Train)')
plt.plot(range(1, len(disc_losses_history) + 1), disc_losses_history, label='Discriminator Loss (Train)')
plt.plot(range(1, len(val_gen_losses_history) + 1), val_gen_losses_history, label='Generator Loss (Validation)')
plt.plot(range(1, len(val_disc_losses_history) + 1), val_disc_losses_history, label='Discriminator Loss (Validation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)



prediction_images = np.load("prediction_images.npy")
prediction_images = prediction_images.reshape(100,200,200)
v_prediction_images = np.vectorize(lambda x: 1 if x > 0.5 else 0)(prediction_images)  
plt.imshow(np.rot90(v_prediction_images[4], k=1) ,cmap="gray")
np.save("v_prediction_images.npy",v_prediction_images)
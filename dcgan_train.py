# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='DCGAN Training with Tiers')
parser.add_argument('--mode', type=str, default='normal', choices=['lite', 'normal', 'heavy'],
                    help='Training tier: lite, normal, or heavy')
args = parser.parse_args()

# --- Config Based on Tier ---
if args.mode == 'lite':
    IMG_SIZE = 64
    MAX_IMAGES = 2000
    EPOCHS = 50
    CLASSIFIER_EPOCHS = 5
    GEN_FILTERS = 256
elif args.mode == 'normal':
    IMG_SIZE = 128
    MAX_IMAGES = 5000
    EPOCHS = 100
    CLASSIFIER_EPOCHS = 10
    GEN_FILTERS = 512
else: # heavy
    IMG_SIZE = 128
    MAX_IMAGES = 10000
    EPOCHS = 300
    CLASSIFIER_EPOCHS = 20
    GEN_FILTERS = 512

print(f"--- Running in {args.mode.upper()} mode ---")
print(f"Resolution: {IMG_SIZE}x{IMG_SIZE}, Epochs: {EPOCHS}, Data: {MAX_IMAGES} images")

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")

DATA_DIR = '/mnt/c/123/data/images'

def load_and_preprocess_images(path):
    images = []
    file_list = os.listdir(path)
    count = 0
    for img_name in file_list:
        if count >= MAX_IMAGES: break
        img = cv2.imread(os.path.join(path, img_name))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = (img.astype(np.float32) / 127.5) - 1.0
        images.append(img)
        count += 1
    return np.array(images)

print("Loading data...")
images = load_and_preprocess_images(DATA_DIR)
print(f"Dataset shape: {images.shape}")

BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices(images)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential()
    # 8x8
    model.add(layers.Dense(8*8*GEN_FILTERS, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, GEN_FILTERS)))
    
    # 16x16
    model.add(layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 32x32
    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 64x64
    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    if IMG_SIZE == 128:
        # 128x128
        model.add(layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh'))
    else:
        # Already at 64, just reduce to 3 channels
        model.pop() # Remove previous ReLU
        model.pop() # Remove previous BN
        model.pop() # Remove previous Conv
        model.add(layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh'))

    return model

def build_discriminator():
    model = tf.keras.Sequential()
    # Input
    model.add(layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    # Second block
    model.add(layers.Conv2D(128, 4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    # Third block
    model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    if IMG_SIZE == 128:
        model.add(layers.Conv2D(512, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0
    plt.figure(figsize=(4,4))
    for i in range(min(16, len(predictions))):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch}")
    output_dir = '/mnt/c/123/epochs_output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()

print("Starting training...")
seed = tf.random.normal([16, 100])
for epoch in range(1, EPOCHS+1):
    for image_batch in dataset:
        gen_loss, disc_loss = train_step(image_batch)
    print(f"PROGRESS: {epoch}/{EPOCHS}")
    if epoch % (10 if EPOCHS > 50 else 5) == 0:
        print(f"Epoch {epoch} | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f}")
        generate_and_save_images(generator, epoch, seed)

gen_path = f'/mnt/c/123/dcgan_generator_{args.mode}.keras'
generator.save(gen_path)
generator.save('/mnt/c/123/dcgan_generator_epoch50.keras') # Compatibility
print(f"Generator saved to {gen_path}")

# --- Augmentation & Classifier ---
print("\n--- Starting Data Augmentation & Classification ---")
csv_path = '/mnt/c/123/HAM10000_metadata.csv'
df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Reload real data for classifier at correct size
X_real, y_real = [], []
malignant = ['mel', 'bcc', 'akiec']
df_meta = pd.read_csv('/mnt/c/123/HAM10000_metadata.csv')
df_meta['label'] = df_meta['dx'].apply(lambda x: 1 if x in malignant else 0)

for _, row in df_meta.head(MAX_IMAGES).iterrows():
    img_path = f'/mnt/c/123/data/images/{row["image_id"]}.jpg'
    if not os.path.exists(img_path): continue
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X_real.append(img/255.0)
    y_real.append(row['label'])

X_real = np.array(X_real)
y_real = np.array(y_real)

# Generate synthetic
NUM_SYNTH = len(X_real) // 2
noise = tf.random.normal([NUM_SYNTH, 100])
X_synth = generator(noise, training=False)
X_synth = (X_synth + 1) / 2.0
y_synth = np.random.choice([0,1], size=len(X_synth))

def build_clf():
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2)
print("Training classifier (Real Only)...")
m1 = build_clf()
h1 = m1.fit(X_train, y_train, epochs=CLASSIFIER_EPOCHS, batch_size=32, validation_data=(X_test, y_test), verbose=0)
acc1 = h1.history['val_accuracy'][-1]

X_aug = np.concatenate([X_real, X_synth])
y_aug = np.concatenate([y_real, y_synth])
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2)
print("Training classifier (Augmented)...")
m2 = build_clf()
h2 = m2.fit(X_train, y_train, epochs=CLASSIFIER_EPOCHS, batch_size=32, validation_data=(X_test, y_test), verbose=0)
acc2 = h2.history['val_accuracy'][-1]

print(f"\nFinal Results ({args.mode}):")
print(f"Real Only Acc: {acc1:.4f}")
print(f"Augmented Acc: {acc2:.4f}")
print(f"Improvement: {(acc2-acc1)*100:.2f}%")
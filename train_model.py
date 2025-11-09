# train_model.py
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

IMG_SIZE = (224,224)
BATCH = 32
EPOCHS_HEAD = 5
EPOCHS_FINETUNE = 5
DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/hotcold_model.keras")   # 👈 notice .keras extension

def make_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR/"train",
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR/"val",
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="binary"
    )
    return train_gen, val_gen

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    # ⚠️ IMPORTANT: remove the preprocessing layer entirely
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, base


if __name__ == "__main__":
    train_gen, val_gen = make_generators()
    model, base = build_model()

    ckpt = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max")
    early = callbacks.EarlyStopping(patience=2, restore_best_weights=True)

    # Phase 1: train top layers
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD, callbacks=[ckpt, early])

    # Phase 2: fine-tune base layers
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINETUNE, callbacks=[ckpt, early])

    # ✅ Save final model in new .keras format (Keras 3 compatible)
    model.save(MODEL_PATH)
    print("✅ Model saved to", MODEL_PATH)

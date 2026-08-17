import tensorflow as tf
import pathlib
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# DATASET PATH
# ==========================================

dataset_path = pathlib.Path(
    r"C:\Users\KOMAL\Documents\Project\RealWaste"
)


# ==========================================
# LOAD DATASET
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print("Classes:", class_names)
print("Number of Classes:", len(class_names))

# ==========================================
# SAVE CLASS NAMES AS .pkl
# ==========================================

with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("Class names saved to class_names.pkl")


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.2),

    tf.keras.layers.RandomZoom(0.2)

])



# Create CNN model
model = tf.keras.Sequential([

    # Input shape:
    
    tf.keras.layers.Input(shape=(224, 224, 3)),

    # -------------------- BLOCK 1 --------------------

    # Create 32 filters of size 3x3
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Reduce image size by half
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # -------------------- BLOCK 2 --------------------

    # Learn more complex features
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # -------------------- BLOCK 3 --------------------

    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # ======================================
    # FLATTEN
    # ======================================

    tf.keras.layers.Flatten(),


    # ======================================
    # DENSE
    # ===============
    # =======================

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.5
    ),


    # ======================================
    # OUTPUT
    # ======================================

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)





# ==========================================
# TRAIN
# ==========================================

history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=30,

)


# ==========================================
# EVALUATE
# ==========================================

test_loss, test_accuracy = model.evaluate(

    test_dataset

)

print(
    "\nFinal Test Accuracy:",
    test_accuracy * 100,
    "%"
)

print(
    "Test Loss:",
    test_loss
)

#  Save Model
model.save("smart_waste_classifier.keras")

print("\nModel Saved Successfully!")
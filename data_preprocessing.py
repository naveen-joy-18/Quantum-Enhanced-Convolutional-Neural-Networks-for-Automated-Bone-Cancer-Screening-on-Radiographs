import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import re

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

def get_dataframe_from_directory(data_dir):
    filepaths = []
    labels = []
    
    # Define a mapping for detailed labels to broader categories
    # This is based on the observed class names from previous runs
    cancerous_keywords = [
        "ewing", "fibrosarcoma", "metastasis", "osteosarcoma", "chondrosarcoma", "encondroma"
    ]
    
    for filename in os.listdir(data_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            filepaths.append(os.path.join(data_dir, filename))
            
            # Extract the relevant part of the filename for classification
            # Remove .rf part and file extension first
            base_name = re.sub(r"\.rf\..*$", "", filename)
            base_name = re.sub(r"\.(jpg|jpeg|png)$", "", base_name, flags=re.IGNORECASE)
            
            # Determine if the image is cancerous or non-cancerous
            is_cancerous = False
            for keyword in cancerous_keywords:
                if keyword in base_name.lower():
                    is_cancerous = True
                    break
            
            if "normal" in base_name.lower():
                labels.append("non_cancerous")
            elif is_cancerous:
                labels.append("cancerous")
            else:
                # For other cases like 'chest_male', 'forearm_male', 'pelvis_other', etc.
                # We'll classify them as non-cancerous for this binary classification task
                # unless they contain a cancerous keyword.
                labels.append("non_cancerous")

    return pd.DataFrame({"filepaths": filepaths, "labels": labels})

def preprocess_data(data_dir):
    train_df = get_dataframe_from_directory(os.path.join(data_dir, "train"))
    valid_df = get_dataframe_from_directory(os.path.join(data_dir, "valid"))
    test_df = get_dataframe_from_directory(os.path.join(data_dir, "test"))

    # Ensure consistent class mapping across generators
    all_labels = ["cancerous", "non_cancerous"]

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=all_labels
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=all_labels
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=all_labels,
        shuffle=False # Keep order for evaluation
    )

    return train_generator, validation_generator, test_generator, len(all_labels)

if __name__ == "__main__":
    data_dir = r"D:\bone_cancer_detector\Images"
    train_data, val_data, test_data, num_classes = preprocess_data(data_dir)
    print("Data preprocessing complete.")
    print(f"Number of training samples: {train_data.samples}")
    print(f"Number of validation samples: {val_data.samples}")
    print(f"Number of test samples: {test_data.samples}")
    print(f"Class names: {train_data.class_indices}")
    print(f"Number of classes: {num_classes}")



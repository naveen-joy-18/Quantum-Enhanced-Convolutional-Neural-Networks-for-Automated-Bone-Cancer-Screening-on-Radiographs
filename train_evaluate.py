import tensorflow as tf
from tensorflow import keras
from model import HQC_model
from data_preprocessing import preprocess_data, IMAGE_SIZE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, RocCurveDisplay
import seaborn as sns
import os

# Create a directory for saving results
results_dir = "training_results"
os.makedirs(results_dir, exist_ok=True)

# Load and preprocess data
data_dir = "bone cancer detection.v1i.multiclass"
train_data, val_data, test_data, num_classes = preprocess_data(data_dir)

print(f"Number of classes: {num_classes}")
print(f"Class names: {train_data.class_indices}")

# Create the model
model = HQC_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=num_classes)

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Store training history for plotting
training_history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

# Custom callback to save model and plots after each epoch
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Store metrics in training history
        training_history["accuracy"].append(logs.get("accuracy", 0))
        training_history["val_accuracy"].append(logs.get("val_accuracy", 0))
        training_history["loss"].append(logs.get("loss", 0))
        training_history["val_loss"].append(logs.get("val_loss", 0))
        
        # Save model after each epoch
        model_path = os.path.join(results_dir, f"bone_cancer_hqc_model_epoch_{epoch+1}.keras")
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Save training history plots after each epoch
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history["accuracy"], label="Training Accuracy")
        plt.plot(training_history["val_accuracy"], label="Validation Accuracy")
        plt.title(f"Model Accuracy - Epoch {epoch+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(training_history["loss"], label="Training Loss")
        plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.title(f"Model Loss - Epoch {epoch+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"training_history_epoch_{epoch+1}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training history plot saved for epoch {epoch+1}")

# Train the model
print("Starting training...")
history = model.fit(
    train_data,
    epochs=100,
    validation_data=val_data,
    verbose=1,
    callbacks=[CustomSaver()]
)

# Evaluate the model on the test set after training completes
print("Evaluating on test data...")
test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions for detailed metrics
print("Generating predictions...")
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_labels = test_data.classes

# Calculate metrics
# For binary classification, roc_auc_score expects the probability of the positive class
# Assuming 'cancerous' is the positive class (index 0 based on previous output)
positive_class_index = train_data.class_indices['cancerous']
auc_roc = roc_auc_score(true_labels, predictions[:, positive_class_index])

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_classes, average="weighted")

print(f"\nDetailed Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# Classification report
class_names = list(train_data.class_indices.keys())
print(f"\nClassification Report:")
print(classification_report(true_labels, predicted_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
plt.close()

# AUC-ROC Plot
plt.figure(figsize=(8, 6))
# For binary classification, RocCurveDisplay.from_predictions is more straightforward
# If num_classes > 2, you'd need to iterate for each class or use a different approach for multi-class ROC
if num_classes == 2:
    # Assuming 'cancerous' is class 0 and 'non_cancerous' is class 1
    # Adjust if your class mapping is different
    positive_class_index = train_data.class_indices['cancerous']
    RocCurveDisplay.from_predictions(true_labels, predictions[:, positive_class_index], name=f"AUC = {auc_roc:.2f}")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    auc_roc_path = os.path.join(results_dir, "auc_roc_curve.svg")
    plt.savefig(auc_roc_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"AUC-ROC curve saved as \"{auc_roc_path}\"")
else:
    print("AUC-ROC curve plotting for multi-class not implemented yet.")

# Final training history plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(training_history["accuracy"], 'b-', label="Training Accuracy", linewidth=2)
plt.plot(training_history["val_accuracy"], 'r-', label="Validation Accuracy", linewidth=2)
plt.title("Model Accuracy Over Time")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(training_history["loss"], 'b-', label="Training Loss", linewidth=2)
plt.plot(training_history["val_loss"], 'r-', label="Validation Loss", linewidth=2)
plt.title("Model Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.bar(class_names, [cm[i, i] for i in range(len(class_names))], color=['lightblue', 'lightcoral'])
plt.title("Correct Predictions by Class")
plt.xlabel("Class")
plt.ylabel("Number of Correct Predictions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_training_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nTraining and evaluation complete!")
print(f"Final Results:")
print(f"- Test Accuracy: {test_accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1-Score: {f1:.4f}")
print(f"- AUC-ROC: {auc_roc:.4f}")
print(f"- Confusion Matrix saved as \"{confusion_matrix_path}\"")
print(f"- Training History plots saved in {results_dir}")
print(f"- Final summary plot saved as \"final_training_summary.png\"")



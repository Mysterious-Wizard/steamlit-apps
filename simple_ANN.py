# --------------------------
# 1. Imports
# --------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------
# 2. Load Dataset
# --------------------------
df = pd.read_csv(r"D:\Coding\Python\AI&ML\ANN\creditcard.csv\creditcard.csv")
print("Dataset shape:", df.shape)
print(df['Class'].value_counts())

# Features & labels
X = df.drop("Class", axis=1).values
y = df["Class"].values

# --------------------------
# 3. Preprocessing
# --------------------------
# Scale 'Amount' and 'Time'
scaler = StandardScaler()
X[:, [0, -1]] = scaler.fit_transform(X[:, [0, -1]])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# --------------------------
# 4. Build Final ANN Model with Best Hyperparameters
# --------------------------
# Best hyperparameters from Trial 3 of the search:
# units_1: 96, dropout_1: 0.4
# num_layers: 3, units_2: 48, dropout_2: 0.3
# units_3: 32, dropout_3: 0.4
# learning_rate: 0.001
model = Sequential([
    Dense(96, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(48, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# --------------------------
# 5. Train Model
# --------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=2048,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# --------------------------
# 6. Evaluate Model
# --------------------------
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_probs))

# --------------------------
# 7. Plot Training Curves
# --------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")

plt.show()

# --------------------------
# 8. Save the Model
# --------------------------
# Save the entire model in the Keras v3 format (.keras)
model.save('credit_card_fraud_model.keras')
print("Model saved successfully as credit_card_fraud_model.keras")

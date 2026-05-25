import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics

os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

# Load Data
df = pd.read_csv('train/train.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_columns = list(df.columns[:-1])

# Save feature columns
with open('artifacts/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
print("Feature columns saved!")

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler and test data
joblib.dump(scaler, 'artifacts/scaler.joblib')
np.save('artifacts/X_test_cnn.npy', X_test)
np.save('artifacts/y_test.npy', y_test)
print("Test data and scaler saved!")

# Build Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test))

# Save model as .keras
model.save('models/model.keras')
print("Model saved as models/model.keras!")

# Save training history
history_dict = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy']
}
with open('artifacts/training_history.json', 'w') as f:
    json.dump(history_dict, f)
print("Training history saved!")

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('artifacts/training_curves.png')
plt.show()

# Predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))


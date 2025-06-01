import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# Paths
DATA_DIR = 'data'  # Folder where your folders A-Z, 0-9 are stored

# Load Data
X = []
y = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_dir):
        for file in os.listdir(label_dir):
            if file.endswith('.npy'):
                data = np.load(os.path.join(label_dir, file))
                X.append(data)
                y.append(label)

X = np.array(X)
y = np.array(y)

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop])

# Save Model and Label Encoder
model.save('models/asl_model_v2.h5')

# Save label classes separately
np.save('models/labels.npy', le.classes_)

print("Model and labels saved successfully!")

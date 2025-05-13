import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import mlflow
import mlflow.keras

# Load the preprocessed data
def load_data():
    # Assuming that you have preprocessed the data and saved it in a .npz file
    data = np.load('data/preprocessed_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Reshape MFCCs to include a single channel for Conv2D input
    X_train = X_train.reshape(-1, 130, 13, 1)  # Add channel dimension (1)
    X_test = X_test.reshape(-1, 130, 13, 1)    # Add channel dimension (1)
    
    return X_train, X_test, y_train, y_test

# Define the CNN model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        # Conv2D layer with input shape (height, width, channels)
        Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer with number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model and log it with MLflow
def train_and_log_model():
    X_train, X_test, y_train, y_test = load_data()
    
    # Get the number of classes
    num_classes = len(np.unique(y_train))
    
    # Initialize the model
    model = create_model(X_train.shape[1:], num_classes)
    
    # Define the MLflow experiment
    mlflow.set_experiment('music_genre_classification')
    
    # Log the model and metrics with MLflow
    with mlflow.start_run():
        # Log model parameters (optional)
        mlflow.log_param('batch_size', 32)
        mlflow.log_param('epochs', 10)
        
        # Set up a ModelCheckpoint callback to save the best model
        checkpoint = ModelCheckpoint('models/music_genre_cnn_model.h5', monitor='val_loss', save_best_only=True)
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])
        
        # Log metrics (accuracy)
        mlflow.log_metric('train_accuracy', history.history['accuracy'][-1])
        mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])
        
        # Save the model in MLflow
        mlflow.keras.log_model(model, 'model')
    
    # Save the model locally
    model.save('models/music_genre_cnn_model.h5')
    print(f"Model saved to: models/music_genre_cnn_model.h5")

# Run the function to train and log the model
if __name__ == "__main__":
    train_and_log_model()

import data_loading
import data_preprocessing
import model
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# Step 1: Load the data
X, y = data_loading.load_data('data/combined_data.csv')

# Step 2: Preprocess the data
X_normalized = data_preprocessing.normalize_data(X)
X_train, X_test, y_train, y_test = data_preprocessing.split_data(X_normalized, y)

# Step 3: Build and compile the model
input_shape = (X_train.shape[1], 1)  # Adjusted input shape
cnn_model = model.build_model(input_shape)

# Reshape the data for Conv1D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

# Train the model with the learning rate scheduler
history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Save the model to a file
cnn_model.save('rainfall_model.h5')

# Step 4: Evaluate the model
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions and compare with true labels
predictions = cnn_model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int).flatten()
comparison = pd.DataFrame({'Predicted': predicted_classes, 'Actual': y_test})
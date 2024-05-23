import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from keras.optimizers import Adam

def build_model(input_shape):
    """
    Build and compile the CNN model with adjustments.

    Args:
    input_shape (tuple): The shape of the input data.

    Returns:
    tf.keras.Model: The compiled CNN model.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
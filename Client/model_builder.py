import tensorflow as tf

def create_DNN(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(input_shape[1:])))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

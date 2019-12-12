import tensorflow as tf, numpy as np


def _input_fn(imgs, labels, batch_size=256, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"imgs": imgs.reshape((len(imgs), 28, 28, 1))}, 
        y=labels,
        batch_size=batch_size, 
        shuffle=shuffle,
    )


def train(train_data, test_data, learning_rate, batch_size, epochs):
    """ 
    Train tf.estimator.DNNClassifier for classifying MNIST digits. 
    
    Parameters
    ----------
    learning_rate: float
        Learning rate, used for training the model.
    batch_size: float
        Batch size, used for training the model.
    epochs: int
        Amount of epochs, during which the model will be trained. 
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    train_imgs, train_labels = train_data
    test_imgs, test_labels = test_data

    train_fn = _input_fn(train_imgs, train_labels, batch_size=batch_size, shuffle=True)
    test_fn = _input_fn(test_imgs, test_labels, batch_size=batch_size, shuffle=True)
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28, 1))
    num_classes = len(np.unique(np.hstack([train_labels, test_labels])))

    estimator = tf.estimator.DNNClassifier(
        n_classes=num_classes,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    
    estimator.train(train_fn)
    estimator.evaluate(test_fn)
    return estimator


def save(estimator):
    serving_input_receiver_fn = tf.estimator.export \
        .build_raw_serving_input_receiver_fn({
            "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28, 1))})
    return estimator.export_saved_model(
        "classifier", serving_input_receiver_fn).decode()
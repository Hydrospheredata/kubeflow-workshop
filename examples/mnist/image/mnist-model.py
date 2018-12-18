import os
import tensorflow as tf
import numpy as np

models_path = os.environ.get("MNIST_MODELS_DIR", "models/mnist")
models_path = os.path.join(models_path, "model")
base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
train_file = "train.npz"
test_file = "t10k.npz"

learning_rate = os.environ.get("LEARNING_RATE", 0.01)
num_steps = os.environ.get("LEARNING_STEPS", 10000)
batch_size = os.environ.get("BATCH_SIZE", 256)


def input_fn(file):
    with np.load(os.path.join(base_path, file)) as data:
        imgs = data["imgs"]
        labels = data["labels"].astype(int)
    return tf.estimator.inputs.numpy_input_fn(
        x = {"imgs": imgs}, y=labels, shuffle=True)

if __name__ == "__main__":
    imgs = tf.feature_column.numeric_column("imgs", shape=(28,28))
    train_fn, test_fn = input_fn(train_file), input_fn(test_file)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[256, 64],
        feature_columns=[imgs],
        n_classes=10,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

    tf.logging.set_verbosity(tf.logging.INFO)
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=num_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28))})
    estimator.export_savedmodel(models_path, serving_input_receiver_fn)

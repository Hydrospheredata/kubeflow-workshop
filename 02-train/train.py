import os, json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

models_path = os.environ.get("MNIST_MODELS_DIR", "models/mnist")
base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
recurring_run = int(os.environ.get("RECURRING_RUN", "0"))

if recurring_run:
    train_file = "subsample-train.npz"
    test_file = "subsample-test.npz"
else: 
    train_file = "train.npz"
    test_file = "t10k.npz"

learning_rate = float(os.environ.get("LEARNING_RATE", 0.01))
num_steps = int(os.environ.get("LEARNING_STEPS", 500))
batch_size = int(os.environ.get("BATCH_SIZE", 256))


def input_fn(file, shuffle=True):
    with np.load(os.path.join(base_path, file)) as data:
        imgs = data["imgs"]
        labels = data["labels"].astype(int)
    return imgs, labels, tf.estimator.inputs.numpy_input_fn(
        x = {"imgs": imgs}, y=labels, shuffle=shuffle, batch_size=batch_size)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare data inputs
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28))
    _, _, train_fn = input_fn(train_file)
    _, labels, test_fn = input_fn(test_file, shuffle=False)

    # Create the model
    estimator = tf.estimator.DNNClassifier(
        n_classes=10,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

    # Train and evaluate the model
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=num_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Export the model 
    serving_input_receiver_fn = tf.estimator \
        .export.build_raw_serving_input_receiver_fn(
            {"imgs": tf.placeholder(tf.float32, shape=(None, 28, 28))})
    estimator.export_savedmodel(models_path, serving_input_receiver_fn)

    # Perform metrics calculations
    accuracy_file = "./accuracy.txt" if dev_env else "/accuracy.txt"
    metrics_file = "./mlpipeline-metrics.json" if dev_env else "/mlpipeline-metrics.json"
    
    accuracy = accuracy_score(labels, list(map(lambda x: x["class_ids"][0], estimator.predict(test_fn))))
    metrics = {
        'metrics': [
            {
                'name': 'accuracy-score',   # -- The name of the metric. Visualized as the column 
                                            # name in the runs table.
                'numberValue':  accuracy,   # -- The value of the metric. Must be a numeric value.
                'format': "PERCENTAGE",     # -- The optional format of the metric. Supported values are 
                                            # "RAW" (displayed in raw format) and "PERCENTAGE" 
                                            # (displayed in percentage format).
            },
        ],
    }

    # Dump metrics
    with open(accuracy_file, "w+") as file:
        file.write(str(accuracy))
    
    with open(metrics_file, "w+") as file:
        json.dump(metrics, file)
    
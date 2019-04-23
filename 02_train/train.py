import os, json, sys
import tensorflow as tf
import numpy as np


mount_path = os.environ.get("MOUNT_PATH", "./")
models_path = os.path.join(mount_path, "models")
data_path = sys.argv[1]
dev_env = int(os.environ.get("DEV_ENV", "0"))

train_file = "train.npz"
test_file = "test.npz"

learning_rate = float(os.environ.get("LEARNING_RATE", 0.01))
epochs = int(os.environ.get("EPOCHS", 10))
batch_size = int(os.environ.get("BATCH_SIZE", 256))


def input_fn(file, shuffle=True):
    with np.load(os.path.join(data_path, file)) as data:
        imgs = data["imgs"]
        labels = data["labels"].astype(int)
    return tf.estimator.inputs.numpy_input_fn(x={"imgs": imgs}, y=labels, 
        shuffle=shuffle, batch_size=batch_size, num_epochs=epochs)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare data inputs
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28))
    train_fn, test_fn = input_fn(train_file), input_fn(test_file)

    # Create the model
    estimator = tf.estimator.DNNClassifier(
        n_classes=10,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

    # Train and evaluate the model
    estimator.train(train_fn)
    evaluation = estimator.evaluate(test_fn)
    accuracy = float(evaluation["accuracy"])

    # Export the model 
    serving_input_receiver_fn = tf.estimator \
        .export.build_raw_serving_input_receiver_fn(
            {"imgs": tf.placeholder(tf.float32, shape=(None, 28, 28))})
    estimator.export_savedmodel(models_path, serving_input_receiver_fn)

    # Perform metrics calculations
    accuracy_file = "./accuracy.txt" if dev_env else "/accuracy.txt"
    metrics_file = "./mlpipeline-metrics.json" if dev_env else "/mlpipeline-metrics.json"
    
    metrics = {
        'metrics': [
            {
                'name': 'accuracy-score',   # -- The name of the metric. Visualized as the column 
                                            # name in the runs table.
                'numberValue': accuracy,    # -- The value of the metric. Must be a numeric value.
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
    
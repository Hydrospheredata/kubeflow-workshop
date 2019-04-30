import os, json, sys
import tensorflow as tf
import numpy as np
import argparse


def input_fn(imgs, labels, batch_size=256, epochs=10):
    return tf.estimator.inputs.numpy_input_fn(
        x={"imgs": imgs}, y=labels, shuffle=True, batch_size=batch_size, num_epochs=epochs)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-path', 
        help='Path, where the current run\'s data was stored',
        required=True)
    parser.add_argument(
        '--mount-path',
        help='Path to PersistentVolumeClaim, deployed on the cluster',
        required=True)
    
    parser.add_argument(
        '--learning-rate', type=float, default=0.01)
    parser.add_argument(
        '--epochs', type=int, default=10)
    parser.add_argument(
        '--batch-size', type=int, default=256)
    parser.add_argument(
        '--dev', help="Flag for development purposes", type=bool, default=False)
    
    args = parser.parse_args()
    arguments = args.__dict__

    models_path = os.path.join(arguments["mount_path"], "models")

    # Prepare data inputs
    with np.load(os.path.join(arguments["data_path"], "train.npz")) as data:
        train_imgs = data["imgs"]
        train_labels = data["labels"].astype(int)
    
    with np.load(os.path.join(arguments["data_path"], "test.npz")) as data:
        test_imgs = data["imgs"]
        test_labels = data["labels"].astype(int)

    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28))
    
    train_fn = input_fn(
        train_imgs, train_labels, 
        batch_size=arguments["batch_size"], 
        epochs=arguments["epochs"])
    
    test_fn = input_fn(
        test_imgs, test_labels,
        batch_size=arguments["batch_size"], 
        epochs=arguments["epochs"])
    
    # Create the model
    estimator = tf.estimator.DNNClassifier(
        n_classes=len(np.unique(np.hstack([train_labels, test_labels]))),
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=arguments["learning_rate"]))

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
    accuracy_file = "./accuracy.txt" if arguments["dev"] else "/accuracy.txt"
    metrics_file = "./mlpipeline-metrics.json" if arguments["dev"] else "/mlpipeline-metrics.json"
    
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
    
import os, json, sys, shutil, tempfile
import tensorflow as tf, numpy as np
import urllib.parse, argparse, mlflow, mlflow.tensorflow
from sklearn.metrics import confusion_matrix
from decouple import Config, RepositoryEnv

from storage import *
from orchestrator import *


config = Config(RepositoryEnv("config.env"))
MLFLOW_LINK = config("MLFLOW_LINK")


def main(data_path, learning_rate, epochs, batch_size, bucket_name, model_name, storage_path="/"):

    # Define helper classes
    storage = Storage(bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Set up environment and variables
    tf.logging.set_verbosity(tf.logging.INFO)
    model_path = os.path.join("model", "mnist")

    # Log params into Mlflow
    mlflow.set_tracking_uri(MLFLOW_LINK)
    mlflow.set_experiment(f'Default.{model_name}')  # Example usage
    mlflow.log_params({
        "data_path": data_path,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    })

    # Download training/testing data
    storage.download_file(os.path.join(data_path, 'test.npz'), './test.npz')
    storage.download_file(os.path.join(data_path, 'train.npz'), './train.npz')

    # Prepare data inputs
    with np.load("./train.npz") as data:
        train_imgs = data["imgs"]
        train_labels = data["labels"].astype(int)
    with np.load("./test.npz") as data:
        test_imgs = data["imgs"]
        test_labels = data["labels"].astype(int)

    train_fn = input_fn(train_imgs, train_labels, batch_size=batch_size, epochs=epochs)
    test_fn = input_fn(test_imgs, test_labels, batch_size=batch_size, epochs=epochs)
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28, 1))
    num_classes = len(np.unique(np.hstack([train_labels, test_labels])))

    # Create the model
    estimator = tf.estimator.DNNClassifier(
        model_dir=model_path,
        n_classes=num_classes,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

    # Train and evaluate the model
    evaluation = estimator.train(train_fn).evaluate(test_fn)
    cm = _calculate_confusion_matrix(test_imgs, test_labels, estimator)

    # Export the model 
    serving_input_receiver_fn = tf.estimator \
        .export.build_raw_serving_input_receiver_fn({
            "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28, 1))})
    saved_model_path = estimator.export_saved_model(
        model_path, serving_input_receiver_fn).decode()

    # mlflow.tensorflow.log_model(
    #     tf_saved_model_dir=saved_model_path,
    #     tf_meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
    #     tf_signature_def_key="predict",
    #     artifact_path=model_path,
    # )
    
    # Prettify folder structure
    final_dir = _bring_folder_structure_in_correct_order(model_path, saved_model_path)
    final_dir_with_prefix = os.path.join(storage.full_name, final_dir)
    
    # Upload files to the cloud
    for root, dirs, files in os.walk(final_dir):
        for file in files:
            source_path = os.path.join(root, file)
            storage.upload_file(source_path, source_path)

    # Upload confusion_matrix to the cloud
    np.savetxt("cm.csv", cm, fmt='%d', delimiter=',')
    cm_cloud_path = storage.upload_file("./cm.csv", os.path.join(final_dir, "cm.csv"))

    # Export metadata to the orchestrator
    metrics = {
        'metrics': [
            {
                'name': 'model-accuracy', 
                'numberValue': evaluation["accuracy"].item(), 
                'format': "PERCENTAGE",    
            },
        ],
    }

    metadata = {
        'outputs': [
            {
                'type': 'tensorboard',
                'source': final_dir_with_prefix,
            },
            {
                'type': 'table',
                'storage': storage.prefix,
                'format': 'csv',
                'source': cm_cloud_path,
                'header': [
                    'one', 'two', 'three', 'four', 'five', 
                    'six', 'seven', 'eight', 'nine', 'ten'
                ],
            }
        ]
    }

    # Collect and export metrics and other parameters
    for key, value in evaluation.items():
        mlflow.log_metric(key, value)
        orchestrator.export_meta(key, value, "txt")

    mlflow.log_param("model_path", os.path.join(storage.full_name, final_dir))
    
    run = mlflow.active_run()
    mlflow_link = f"{MLFLOW_LINK}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    orchestrator.export_meta("mlpipeline-metrics", metrics, "json")
    orchestrator.export_meta("mlpipeline-ui-metadata", metadata, "json")
    orchestrator.export_meta("model_path", final_dir_with_prefix, "txt")
    orchestrator.export_meta("classes", num_classes, "txt")
    orchestrator.export_meta("mlflow_link", mlflow_link, "txt")


def input_fn(imgs, labels, batch_size=256, epochs=10, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"imgs": imgs.reshape((len(imgs), 28, 28, 1))}, 
        y=labels, shuffle=shuffle, batch_size=batch_size, num_epochs=epochs)


def _calculate_confusion_matrix(imgs, labels, model):
    cm_fn = input_fn(imgs=imgs, labels=labels, epochs=1, shuffle=False)
    result = list(map(lambda x: x["class_ids"][0], model.predict(cm_fn)))
    return confusion_matrix(labels, result)


def _bring_folder_structure_in_correct_order(all_models_path, current_model_path):
    """ Bring folder structure in correct order, so it can be safely 
        uploaded to the cloud. """

    def _relative_files_move(from_dir, to_dir):
        for root, dirs, files in os.walk(from_dir):
            for file in files:
                relpath = os.path.relpath(root, from_dir)
                os.makedirs(os.path.join(to_dir, relpath), exist_ok=True)
                shutil.move(os.path.join(root, file), os.path.join(to_dir, relpath, file)) 

    timestamp = os.path.basename(current_model_path)
    final_dir = os.path.join(all_models_path, timestamp)
    saved_model_dir = os.path.join(final_dir, "saved_model")
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        _relative_files_move(current_model_path, tmpdir1)
        shutil.rmtree(current_model_path)

        _relative_files_move(all_models_path, tmpdir2)
        shutil.rmtree(all_models_path)
        
        os.makedirs(final_dir)
        os.makedirs(saved_model_dir)

        _relative_files_move(tmpdir1, saved_model_dir)
        _relative_files_move(tmpdir2, final_dir)
    
    return final_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', help='Path, where the current run\'s data was stored', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=3500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        bucket_name=args.bucket_name,
    )
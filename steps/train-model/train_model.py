import os, json, sys, shutil, tempfile, logging, re
import tensorflow as tf, numpy as np
import urllib.parse, argparse, mlflow, mlflow.tensorflow
from sklearn.metrics import confusion_matrix
from cloud import CloudHelper


logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("train_model.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def input_fn(imgs, labels, batch_size=256, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"imgs": imgs.reshape((len(imgs), 28, 28, 1))}, 
        y=labels,
        batch_size=batch_size, 
        shuffle=shuffle,
    )


def _calculate_confusion_matrix(imgs, labels, model):
    cm_fn = input_fn(imgs=imgs, labels=labels, shuffle=False)
    result = list(map(lambda x: x["class_ids"][0], model.predict(cm_fn)))
    return confusion_matrix(labels, result)


def _prettify_folder_structure(model_dir, export_model_path):
    """ 
    Parameters
    ----------
    model_dir: str
        A directory, where all Tensorflow checkpoints are saved.
    export_model_path: str
        A directory, where current model is exported in saved_model format. 
    """

    def _relative_files_move(from_dir, to_dir):
        for root, dirs, files in os.walk(from_dir):
            for file in files:
                relpath = os.path.relpath(root, from_dir)
                os.makedirs(os.path.join(to_dir, relpath), exist_ok=True)
                shutil.move(os.path.join(root, file), os.path.join(to_dir, relpath, file)) 

    final_dir = model_dir
    saved_model_dir = os.path.join(final_dir, "saved_model")
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        _relative_files_move(export_model_path, tmpdir1)
        shutil.rmtree(export_model_path)

        _relative_files_move(model_dir, tmpdir2)
        shutil.rmtree(model_dir)
        
        os.makedirs(final_dir)
        os.makedirs(saved_model_dir)

        _relative_files_move(tmpdir1, saved_model_dir)
        _relative_files_move(tmpdir2, final_dir)
    
    return final_dir


def main(data_path, model_path, learning_rate, batch_size, epochs, *args, **kwargs):
    """ 
    Train tf.estimator.DNNClassifier for classifying MNIST digits. 
    
    Parameters
    ----------
    data_path: str
        Path to directory, where training data is located.
    model_path: str
        Path to directory, where model should be exported.
    learning_rate: float
        Learning rate, used for training the model.
    batch_size: float
        Batch size, used for training the model.
    epochs: int
        Amount of epochs, during which the model will be trained. 
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare data inputs
    with np.load(os.path.join(data_path, "train", "imgs.npz")) as np_imgs:
        train_imgs = np_imgs["imgs"]
    with np.load(os.path.join(data_path, "train", "labels.npz")) as np_labels:
        train_labels = np_labels["labels"].astype(int)
    with np.load(os.path.join(data_path, "t10k", "imgs.npz")) as np_imgs:
        test_imgs = np_imgs["imgs"]
    with np.load(os.path.join(data_path, "t10k", "labels.npz")) as np_labels:
        test_labels = np_labels["labels"].astype(int)

    train_fn = input_fn(train_imgs, train_labels, batch_size=batch_size, shuffle=True)
    test_fn = input_fn(test_imgs, test_labels, batch_size=batch_size, shuffle=True)
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28, 1))
    num_classes = len(np.unique(np.hstack([train_labels, test_labels])))

    # Create the model
    estimator = tf.estimator.DNNClassifier(
        model_dir=model_path,
        n_classes=num_classes,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
    )
    
    # Train and evaluate the model
    evaluation = estimator.train(train_fn).evaluate(test_fn)
    cm = _calculate_confusion_matrix(test_imgs, test_labels, estimator)
    np.savetxt(os.path.join(model_path, "cm.csv"), cm, fmt='%d', delimiter=',')

    # Export the model 
    serving_input_receiver_fn = tf.estimator.export \
        .build_raw_serving_input_receiver_fn({
            "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28, 1))})
    saved_model_path = estimator.export_saved_model(
        model_path, serving_input_receiver_fn).decode()
    _prettify_folder_structure(model_path, saved_model_path)

    evaluation.update({"num_classes": num_classes})
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dev', action='store_true', default=False)
    args = parser.parse_args()

    # Prepare working environment 
    cloud = CloudHelper(default_config_map_params={"uri.mlflow": "http://mlflow.k8s.hydrosphere.io"})
    cloud.set_mlflow_endpoint(cloud.get_kube_config_map()["uri.mlflow"])
    cloud.set_mlflow_experiment(f"Default.{args.model_name}")
    
    kwargs = dict(vars(args)) # create a copy of the args dict, since we are going to mutate it
    data_version = re.findall(r'sample-version=(\w+)', args.data_path)[0]
    model_version = CloudHelper._md5_string("".join(sys.argv))
    full_model_path = os.path.join(args.model_path, args.model_name, 
        f"data-version={data_version}", f"model-version={model_version}")

    # Download training data
    cloud.download_prefix(args.data_path, args.data_path)

    # Run training
    kwargs["data_path"] = cloud.get_relative_path_from_uri(args.data_path)
    kwargs["model_path"] = cloud.get_relative_path_from_uri(args.model_path)
    result = main(**kwargs)

    # Upload artifacts
    cloud.upload_prefix(kwargs["model_path"], full_model_path)

    # Log execution
    # MLflow
    params = vars(args)
    params.update({"model_path": full_model_path})
    # Kubeflow
    outputs = {
        "mlpipeline-metrics.json": {       # mlpipeline-metrics.json lets us enrich Kubeflow UI
            "metrics": [
                {
                    "name": "accuracy",
                    "numberValue": result["accuracy"].item(),
                    "format": "PERCENTAGE"
                },
                {
                    "name": "loss",
                    "numberValue": result["average_loss"].item(),
                    "format": "RAW"
                }
            ]
        },
        "mlpipeline-ui-metadata.json": {    # mlpipeline-ui-metadata.json lets us enrich Artifacts section of the ComponentOp
            'outputs': [
                {
                    'type': 'tensorboard',
                    'source': full_model_path,
                },
                {
                    'type': 'table',
                    'storage': cloud.get_bucket_from_uri(args.model_path).scheme,
                    'format': 'csv',
                    'source': os.path.join(full_model_path, "cm.csv"),
                    'header': [
                        'one', 'two', 'three', 'four', 'five', 
                        'six', 'seven', 'eight', 'nine', 'ten'
                    ],
                }
            ]
        },
        "model_path": full_model_path,
    }
    outputs.update(result)
    cloud.log_execution(
        params=params, 
        outputs=outputs, 
        logs_path="mnist/logs", 
        logs_file="train_model.log",
        logs_bucket=cloud.get_bucket_from_uri(args.data_path).full_uri,
        dev=args.dev,
    )

    # # Model artifacts can also be exported into MLFlow instance
    # mlflow.tensorflow.log_model(
    #     tf_saved_model_dir=saved_model_path,
    #     tf_meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
    #     tf_signature_def_key="predict",
    #     artifact_path=model_path,
    # )

    # run = mlflow.active_run()
    # mlflow_run_uri = f"{mlflow_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    # cloud.export_metas({
    #     "mlflow_run_uri": mlflow_run_uri,
    # })
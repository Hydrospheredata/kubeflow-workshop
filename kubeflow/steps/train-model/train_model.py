import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_train_model.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/step_train_model.log")])
logger = logging.getLogger(__name__)

import os, json, shutil, tempfile, re
import tensorflow as tf, numpy as np, wo
import urllib.parse, argparse, mlflow, mlflow.tensorflow
from sklearn.metrics import confusion_matrix


INPUTS_DIR, OUTPUTS_DIR = "inputs/", "outputs/"
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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


def main(data_path, model_path, learning_rate, batch_size, epochs, full_model_path, *args, **kwargs):
    """ 
    Train tf.estimator.DNNClassifier for classifying MNIST digits. 
    
    Parameters
    ----------
    data_path: str
        Path to directory, where training data is located.
    model_path: str
        Path to directory, where model should be exported.
    full_model_path: str
        Local path with respect to sample/model version, where model 
        artifacts will be stored.
    learning_rate: float
        Learning rate, used for training the model.
    batch_size: float
        Batch size, used for training the model.
    epochs: int
        Amount of epochs, during which the model will be trained. 
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare data inputs
    with np.load(os.path.join(INPUTS_DIR, "train", "imgs.npz")) as np_imgs:
        train_imgs = np_imgs["imgs"]
    with np.load(os.path.join(INPUTS_DIR, "train", "labels.npz")) as np_labels:
        train_labels = np_labels["labels"].astype(int)
    with np.load(os.path.join(INPUTS_DIR, "t10k", "imgs.npz")) as np_imgs:
        test_imgs = np_imgs["imgs"]
    with np.load(os.path.join(INPUTS_DIR, "t10k", "labels.npz")) as np_labels:
        test_labels = np_labels["labels"].astype(int)

    train_fn = input_fn(train_imgs, train_labels, batch_size=batch_size, shuffle=True)
    test_fn = input_fn(test_imgs, test_labels, batch_size=batch_size, shuffle=True)
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28, 1))
    num_classes = len(np.unique(np.hstack([train_labels, test_labels])))

    # Create the model
    estimator = tf.estimator.DNNClassifier(
        model_dir=full_model_path,
        n_classes=num_classes,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
    )
    
    # Train and evaluate the model
    evaluation = estimator.train(train_fn).evaluate(test_fn)
    cm = _calculate_confusion_matrix(test_imgs, test_labels, estimator)
    np.savetxt(os.path.join(full_model_path, "cm.csv"), cm, fmt='%d', delimiter=',')

    # Export the model 
    serving_input_receiver_fn = tf.estimator.export \
        .build_raw_serving_input_receiver_fn({
            "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28, 1))})
    saved_model_path = estimator.export_saved_model(
        full_model_path, serving_input_receiver_fn).decode()
    _prettify_folder_structure(full_model_path, saved_model_path)

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
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    inputs = [(args.data_path, INPUTS_DIR)]
    outputs = [(OUTPUTS_DIR, args.model_path)]
    logs_bucket = wo.parse_bucket(args.data_path, with_scheme=True)
    experiment = "MNIST"
    params = {"uri.mlflow": "http://mlflow.eks.hydrosphere.io"}


    with wo.Orchestrator(inputs=inputs, outputs=outputs,
        logs_file=logs_file, logs_bucket=logs_bucket,
        experiment=experiment, default_params=params,
        mlflow=True, dev=args.dev) as w:

        # Initialize runtime variables
        sample_version = re.findall(r'sample-version=(\w+)', args.data_path)[0]
        model_version = wo.utils.io.md5_string("".join(sys.argv))
        model_path = os.path.join(args.model_name, 
            f"sample-version={sample_version}", f"model-version={model_version}")

        # Execute main script
        result = main(**vars(args), full_model_path=os.path.join(OUTPUTS_DIR, model_path))

        parameters = vars(args).copy()
        parameters['model_path'] = model_path
        metrics = result.copy()
        del metrics['num_classes']

        w.log_execution(
            parameters=parameters,
            metrics=metrics,
            outputs={
                "accuracy": result["accuracy"].item(),
                "num_classes": result["num_classes"],
                "average_loss": result["average_loss"].item(),
                "global_step": result["global_step"].item(),
                "loss": result["loss"].item(),
                "model_path": os.path.join(args.model_path, model_path), 
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
                    'version': 1, 
                    'outputs': [
                        {
                            'type': 'tensorboard',
                            'source': os.path.join(args.model_path, model_path),
                        },
                        {
                            'type': 'table',
                            'storage': 's3',
                            'format': 'csv',
                            'source': os.path.join(args.model_path, model_path, "cm.csv"),
                            'header': [
                                'one', 'two', 'three', 'four', 'five', 
                                'six', 'seven', 'eight', 'nine', 'ten'
                            ],
                        }
                    ]
                },
            },
        )

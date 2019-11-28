import logging, sys, os

os.makedirs("logs", exist_ok=True)
logs_file = "logs/step_train_drift_detector.log"
logging.basicConfig(level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s.%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logs_file)])
logger = logging.getLogger(__name__)

import os, argparse, shutil, urllib
import datetime, re, hashlib
import numpy as np, tensorflow as tf, wo

INPUTS_DIR, OUTPUTS_DIR = "inputs", "outputs"
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def encoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def main(data_path, model_path, learning_rate, batch_size, steps, full_model_path, *args, **kwargs): 
    """ 
    Train pure Tensorflow Autoencoder and upload it to the cloud. 
    
    Parameters
    ----------
    data_path: str
        Local path, where training data is stored. 
    learning_rate: float
        Learning rate, used to train the model.
    batch_size: int
        How much images will be feed at once to the model.
    steps: int
        Amount of steps model training process will take.
    model_path: str
        Local path, where model artifacts will be stored. 
    full_model_path: str
        Local path with respect to sample/model version, where model 
        artifacts will be stored. 
    """

    # Define network parameters
    num_hidden_1 = 256 
    num_hidden_2 = 128 
    num_input = 784

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    }

    # Prepare data inputs
    with np.load(os.path.join(INPUTS_DIR, "train", "imgs.npz")) as data:
        train_imgs = data["imgs"]
    with np.load(os.path.join(INPUTS_DIR, "train", "labels.npz")) as data:
        train_labels = data["labels"]
    with np.load(os.path.join(INPUTS_DIR, "t10k", "imgs.npz")) as data:
        test_imgs = data["imgs"]
    with np.load(os.path.join(INPUTS_DIR, "t10k", "labels.npz")) as data:
        test_labels = data["labels"]

    imgs = np.expand_dims(np.vstack([train_imgs, test_imgs]), axis=-1)
    labels = np.hstack([train_labels, test_labels])
    num_classes = len(np.unique(labels))

    features_placeholder = tf.placeholder(imgs.dtype, imgs.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,))
    dataset = dataset.batch(batch_size).repeat()

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Define layers 
    imgs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    imgs_flattened = tf.layers.flatten(imgs_placeholder)
    encoder_op = encoder(imgs_flattened, weights, biases)
    decoder_op = decoder(encoder_op, weights, biases)

    # Define optimizer, loss function
    y_pred, y_true = decoder_op, imgs_flattened
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=-1)
    score = tf.cast(tf.expand_dims(loss, -1), tf.float64) 
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Run training 
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={features_placeholder: imgs})
        sess.run(tf.global_variables_initializer())

        # Training
        for i in range(1, steps+1):
            batch = sess.run(next_element)[0]
            _, l = sess.run([optimizer, loss], feed_dict={imgs_placeholder: batch})
            
            try:  # in case, MLFlow instance is not available
                if i % 10 == 0: w.log_execution(metrics={"loss": np.mean(l)})
            except: 
                continue

        # Save model
        signature_map = {
            "predict": tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={
                    "imgs": imgs_placeholder,
                    "probabilities": tf.placeholder(dtype=tf.float32, shape=(None, num_classes)),
                    "class_ids": tf.placeholder(dtype=tf.int64, shape=(None, 1)),
                    "logits": tf.placeholder(dtype=tf.float32, shape=(None, num_classes)),
                    "classes": tf.placeholder(dtype=tf.string, shape=(None, 1)),
                }, 
                outputs={"value": score})
        }

        shutil.rmtree(full_model_path, ignore_errors=True) # in case folder exists 
        builder = tf.saved_model.builder.SavedModelBuilder(full_model_path)
        builder.add_meta_graph_and_variables(
            sess=sess, 
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_map)
        builder.save()
    
    return {
        "loss": float(np.mean(l)),
    }


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--dev', action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    if unknown: 
        logger.warning(f"Parsed unknown args: {unknown}")

    inputs = [(args.data_path, INPUTS_DIR)]
    outputs = [(OUTPUTS_DIR, args.model_path)]
    logs_bucket = wo.parse_bucket(args.data_path, with_scheme=True)
    experiment = "MNIST Drift Detector"
    params = {"uri.mlflow": "http://mlflow.eks.hydrosphere.io"}

    with wo.Orchestrator(inputs=inputs, outputs=outputs,
        logs_file=logs_file, logs_bucket=logs_bucket,
        experiment=experiment, default_params=params,
        mlflow=True, dev=args.dev) as w:

        # Initialize runtime variables
        data_version = re.findall(r'sample-version=(\w+)', args.data_path)[0]
        model_version = wo.utils.io.md5_string("".join(sys.argv))
        model_path = os.path.join(args.model_name, 
            f"sample-version={data_version}", f"model-version={model_version}")

        # Execute main script
        result = main(**vars(args), full_model_path=os.path.join(OUTPUTS_DIR, model_path))

        parameters = vars(args).copy()
        parameters['model_path'] = model_path

        # Execution logging
        w.log_execution(
            parameters=parameters,
            outputs={
                "model_path": os.path.join(args.model_path, model_path),
                "mlpipeline-metrics.json": {       # mlpipeline-metrics.json lets us enrich Kubeflow UI
                    "metrics": [
                        {
                            "name": "loss",
                            "numberValue": result["loss"],
                            "format": "RAW"
                        }
                    ]
                },
                "loss": result["loss"],
            },
        )

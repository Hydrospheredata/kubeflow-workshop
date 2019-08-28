import os, argparse, shutil, urllib
import logging, datetime, re, hashlib, sys
import numpy as np, tensorflow as tf, mlflow
from cloud import CloudHelper


logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("train_drift_detector.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def encoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def main(data_path, model_path, learning_rate, batch_size, steps, *args, **kwargs): 
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
    with np.load(os.path.join(data_path, "train", "imgs.npz")) as data:
        train_imgs = data["imgs"]
    with np.load(os.path.join(data_path, "train", "labels.npz")) as data:
        train_labels = data["labels"]
    with np.load(os.path.join(data_path, "t10k", "imgs.npz")) as data:
        test_imgs = data["imgs"]
    with np.load(os.path.join(data_path, "t10k", "labels.npz")) as data:
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
            
            try:
                if i % 10 == 0: cloud.log_metrics({"loss": np.mean(l)})
                if i % 250 == 0 or i == 1: logger.info(f'Step {i}: Loss: {np.mean(l)}')
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
                outputs={"score": score})
        }

        shutil.rmtree(model_path, ignore_errors=True) # in case folder exists 
        builder = tf.saved_model.builder.SavedModelBuilder(model_path)
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
    args = parser.parse_args()
    
    # Prepare working environment 
    cloud = CloudHelper(default_config_map_params={"uri.mlflow": "http://mlflow.k8s.hydrosphere.io"})
    cloud.set_mlflow_endpoint(cloud.get_kube_config_map()["uri.mlflow"])
    cloud.set_mlflow_experiment(f"Default.{args.model_name}")

    kwargs = dict(vars(args))  # create a copy of the args dict, since we are going to mutate it
    data_version = re.findall(r'sample-version=(\w+)', args.data_path)[0]
    model_version = CloudHelper._md5_string("".join(sys.argv))
    full_model_path = os.path.join(args.model_path, args.model_name, 
        f"data-version={data_version}", f"model-version={model_version}")
    
    # Download training data
    cloud.download_prefix(args.data_path, args.data_path)

    # Run training
    kwargs["data_path"] = cloud.get_relative_path_from_uri(args.data_path)
    kwargs["model_path"] = cloud.get_relative_path_from_uri(args.model_path)
    result = main(**kwargs, cloud=cloud)

    # Upload artifacts
    cloud.upload_prefix(kwargs["model_path"], full_model_path)
    
    # Log execution
    params = vars(args)
    params.update({"model_path": full_model_path})
    outputs = {
        "mlpipeline-metrics.json": {       # mlpipeline-metrics.json lets us enrich Kubeflow UI
            "metrics": [
                {
                    "name": "loss",
                    "numberValue": result["loss"],
                    "format": "RAW"
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
        logs_file="train_drift_detector.log",
        logs_bucket=cloud.get_bucket_from_uri(args.data_path).full_uri,
        dev=args.dev,
    )
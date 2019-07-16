import os, argparse
import shutil, urllib, datetime
import numpy as np
import tensorflow as tf
import mlflow
from decouple import Config, RepositoryEnv

from storage import *
from orchestrator import *


config = Config(RepositoryEnv("config.env"))
MLFLOW_LINK = config("MLFLOW_LINK")


def encoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def main(data_path, learning_rate, steps, batch_size, model_name, bucket_name, storage_path="/"): 

    # Define helper classes
    storage = Storage(bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Set up environment and variables
    model_path = os.path.join("model", "mnist-drift-detector", str(round(datetime.datetime.now().timestamp())))

    # Log params into Mlflow
    mlflow.set_tracking_uri(MLFLOW_LINK)
    mlflow.set_experiment(f'Default.{model_name}')  # Example usage
    mlflow.log_params({
        "data_path": data_path,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "steps": steps
    })

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
    
    # Download training/testing data
    storage.download_file(os.path.join(data_path, "train.npz"), "./train.npz")
    storage.download_file(os.path.join(data_path, "test.npz"), "./test.npz")

    # Prepare data inputs
    with np.load("./train.npz") as data:
        train_imgs = data["imgs"]
        train_labels = data["labels"]

    with np.load("./test.npz") as data:
        test_imgs = data["imgs"]
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

            if i % 10 == 0:
                mlflow.log_metric("loss", np.mean(l))
            if i % 500 == 0 or i == 1:
                print(f'Step {i}: Minibatch Loss: {np.mean(l)}', flush=True)

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

    # Upload files to the cloud
    for root, dirs, files in os.walk(model_path):
        for file in files:
            source_path = os.path.join(root, file)
            storage.upload_file(source_path, source_path)

    # Export metadata to the orchestrator
    metrics = {
        'metrics': [
            {
                'name': 'drift-detector-loss',
                'numberValue': float(np.mean(l)), 
                'format': "RAW",                
            },
        ],
    }

    mlflow.log_param("model_path", os.path.join(storage.full_name, model_path))
    
    run = mlflow.active_run()
    mlflow_link = f"{MLFLOW_LINK}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    orchestrator.export_meta("mlpipeline-metrics", metrics, "json")
    orchestrator.export_meta("model_path", os.path.join(storage.full_name, model_path), "txt")
    orchestrator.export_meta("loss", np.mean(l), "txt")
    orchestrator.export_meta("classes", num_classes, "txt")
    orchestrator.export_meta("mlflow_link", mlflow_link, "txt")


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        model_name=args.model_name,
        bucket_name=args.bucket_name,
    )
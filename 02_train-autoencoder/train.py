import os, argparse
import shutil
import numpy as np
import tensorflow as tf
import boto3, urllib
import datetime


# Training Parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path', help='Path, where the current run\'s data was stored', required=True)
parser.add_argument('--hydrosphere-address', required=True)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument(
    '--dev', help='Flag for development purposes', action="store_true")

args = parser.parse_args()
s3 = boto3.resource('s3')

namespace = urllib.parse.urlparse(args.hydrosphere_address).netloc.split(".")[0]
models_path = os.path.join(namespace, "models", "mnist-autoencoder", str(round(datetime.datetime.now().timestamp())))


# Network Parameters
num_hidden_1 = 256 
num_hidden_2 = 128 
num_input = 784


# Download training/testing data
s3.Object('odsc-workshop', os.path.join(args.data_path, "train.npz")).download_file('./train.npz')
s3.Object('odsc-workshop', os.path.join(args.data_path, "test.npz")).download_file('./test.npz')

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
dataset = dataset.batch(args.batch_size).repeat()
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


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


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


imgs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
imgs_flattened = tf.layers.flatten(imgs_placeholder)
encoder_op = encoder(imgs_flattened)
decoder_op = decoder(encoder_op)

y_pred, y_true = decoder_op, imgs_flattened
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=-1)
score = tf.cast(tf.expand_dims(loss, -1), tf.float64) 
optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: imgs})
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(1, args.steps+1):
        batch = sess.run(next_element)[0]
        _, l = sess.run([optimizer, loss], 
            feed_dict={imgs_placeholder: batch})

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

    shutil.rmtree(models_path, ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder(models_path)
    builder.add_meta_graph_and_variables(
        sess=sess, 
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map)
    builder.save()

# Upload model to S3
for root, dirs, files in os.walk(models_path):
    for file in files:
        print(f"Uploading {file} to S3", flush=True)

        location = os.path.join(root, file)
        s3.meta.client.upload_file(location, "odsc-workshop", location)

with open("./model_path.txt" if args.dev else "/model_path.txt", "w+") as file:
    file.write(models_path)

with open("./loss.txt" if args.dev else "/loss.txt", "w+") as file:
    file.write(str(np.mean(l)))

with open("./classes.txt" if args.dev else "/classes.txt", "w+") as file:
    file.write(str(num_classes))
import os
import shutil
import numpy as np
import tensorflow as tf

learning_rate = os.environ.get("LEARNING_RATE", 0.01)
num_steps = os.environ.get("LEARNING_STEPS", 10000)
batch_size = os.environ.get("BATCH_SIZE", 246)
display_step = os.environ.get("DISPLAY_STEPS", 1000)

models_path = os.environ.get("MNIST_MODELS_DIR", "models/mnist")
models_path = os.path.join(models_path, "concept")
base_path = os.environ.get("MNIST_DATA_DIR", "data/mnist")
train_file = "train.npz"

num_hidden_1 = 256 
num_hidden_2 = 128 
num_input = 784

# Import MNIST data
with np.load(os.path.join(base_path, train_file)) as data:
    imgs_data, labels_data = data["imgs"], data["labels"]
    assert imgs_data.shape[0] == labels_data.shape[0]

imgs_placeholder = tf.placeholder(imgs_data.dtype, imgs_data.shape)
labels_placeholder = tf.placeholder(labels_data.dtype, labels_data.shape)

dataset = tf.data.Dataset.from_tensor_slices((imgs_placeholder, labels_placeholder))
dataset = dataset.batch(batch_size).repeat()
iterator = dataset.make_initializable_iterator()
imgs, labels = iterator.get_next()


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

imgs_flattened = tf.layers.flatten(imgs)
encoder_op = encoder(imgs_flattened)
decoder_op = decoder(encoder_op)

y_pred, y_true = decoder_op, imgs_flattened
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2), axis=-1)
serving = tf.expand_dims(loss, 0)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={
        imgs_placeholder: imgs_data,
        labels_placeholder: labels_data})
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(1, num_steps+1):
        _, l = sess.run([optimizer, loss])
        if i % display_step == 0 or i == 1:
            print(f'Step {i}: Minibatch Loss: {np.mean(l)}', flush=True)

    # Save model
    signature_map = {
        "infer": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"X": imgs_placeholder}, 
            outputs={"reconstructed": serving})
    }

    shutil.rmtree(models_path, ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder(models_path)
    builder.add_meta_graph_and_variables(
        sess=sess, 
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map)
    builder.save()
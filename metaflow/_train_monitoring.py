import datetime
import numpy as np, tensorflow as tf, wo


def encoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def train(train_data, test_data, learning_rate, batch_size, steps): 
    """ 
    Train pure Tensorflow Autoencoder.
    
    Parameters
    ----------
    learning_rate: float
        Learning rate, used to train the model.
    batch_size: int
        How much images will be feed at once to the model.
    steps: int
        Amount of steps model training process will take.
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
    train_imgs, train_labels = train_data
    test_imgs, test_labels = test_data

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

        model_path = f"monitoring/{round(datetime.datetime.now().timestamp())}"
        builder = tf.saved_model.builder.SavedModelBuilder(model_path)
        builder.add_meta_graph_and_variables(
            sess=sess, 
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_map)
        builder.save()
    
    return model_path
import numpy as np
import tensorflow as tf

import model_def

train_steps   = 10
seq_length    = 32
batch_size    = 128
data_dim      = 24
learning_rate = 0.01
num_epochs    = 50
lstm_units    = 256
num_labels = 4

trn_data   = np.random.random((batch_size, seq_length, data_dim))
trn_labels = np.random.random((batch_size, num_labels))
val_data   = np.random.random((batch_size, seq_length, data_dim))
val_labels = np.random.random((batch_size, num_labels))

num_trn_examples = trn_labels.shape[0]
num_val_examples = val_labels.shape[0]

x, y, rnn, rnn_state, rnn_final_state, logits, loss, dropout_keep_prob = model_def.model(dropout_keep=0.5)

################################################################################
################################ TRAIN OPS #####################################
################################################################################

global_step = tf.Variable(0, trainable=False)

optimizer = tf.train.RMSPropOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)

# Gradient clipping
grads, variables = zip(*grads_and_vars)
grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
grads_and_vars = zip(grads_clipped, variables)

train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Create a saver for all variables
tf_vars_to_save = tf.trainable_variables() + [global_step]
saver = tf.train.Saver(tf_vars_to_save, max_to_keep=5)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

################################################################################
################################ TRAIN LOOP ####################################
################################################################################

index_in_epoch    = 0
batches_per_epoch = int(np.floor_divide(num_trn_examples, batch_size))


for _ in range(train_steps):

    start = index_in_epoch*batch_size
    end   = start+batch_size

    feed_dict = {
        x: trn_data[start:end,],
        y: trn_labels[start:end,]
    }

    _, step, train_loss = sess.run([train_op, global_step, loss], feed_dict=feed_dict)
    index_in_epoch += 1

    if step % 20 == 0:
        print("Step %05i, Train Loss = %.3f" % (step, train_loss))

    if step % batches_per_epoch == 0:

        # Check performance on validations set
        print("Testing model performance on validation set:")
        num_val_batches = int(np.floor_divide(num_trn_examples, float(batch_size)))
        val_losses     = np.zeros(num_val_batches, dtype=np.float32)
        val_accuracies = np.zeros(num_val_batches, dtype=np.float32)

        for i in range(num_val_batches):

            start = i * batch_size
            end = start + batch_size

            feed_dict = {
                x: val_data[start:end, ],
                y: val_labels[start:end, ]
            }

            val_loss = sess.run(loss, feed_dict=feed_dict)
            # print("  %03i. Validation Accuracy = %.2f, Validation Loss = %.3f" % (i, val_acc, val_loss))

            val_losses[i]     = val_loss

        print("  Average Validation Accuracy: %.3f" % np.mean(val_accuracies))
        print("  Average Validation Loss:     %.3f" % np.mean(val_losses))

        # Shuffling training data for next epoch
        perm = np.arange(len(trn_data))
        np.random.shuffle(perm)
        trn_data = trn_data[perm]
        trn_labels = trn_labels[perm]
        index_in_epoch = 0

        # Save the checkpoint to disk
        path = saver.save(sess, "./checkpoints/chkpt", global_step=global_step)
        print("Checkpoint saved to %s" % path)

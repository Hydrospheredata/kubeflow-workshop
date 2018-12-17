import tensorflow as tf


def model(
        seq_length=32,
        batch_size=128,
        data_dim=24,
        lstm_units=256,
        num_labels=4,
        dropout_keep=1.0
):
    x = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    y = tf.placeholder(tf.float32, [None, 4])
    dropout_keep_prob = tf.constant(
        dropout_keep,
        dtype=tf.float32)

    # Note: using the LSTMBlockCell
    rnn_cell = tf.contrib.rnn.LSTMBlockCell(lstm_units, forget_bias=1.0)
    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep_prob)
    rnn = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * 1)  # single layer
    rnn_init_state = rnn.zero_state(batch_size, tf.float32)

    weight_init = tf.contrib.layers.variance_scaling_initializer()
    output_W = tf.get_variable("W", shape=[lstm_units, num_labels], initializer=weight_init)
    output_b = tf.get_variable("b", shape=[num_labels], initializer=tf.constant_initializer(0.0))

    # Split into timesteps
    x_split = tf.split(x, seq_length, 1)
    h_rnn = None
    rnn_state = None
    # LSTM unrolling for timesteps
    for step in range(seq_length):
        with tf.variable_scope("RNN") as scope:
            if step > 0:
                scope.reuse_variables()
            input_step = tf.squeeze(x_split[step], [1])
            h_rnn, rnn_state = rnn(input_step, rnn_init_state)

    rnn_final_state = rnn_state

    # Outputs
    logits = tf.matmul(h_rnn, output_W) + output_b

    # Compute batch loss
    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(loss_per_example)
    return x, y, rnn, rnn_init_state, rnn_final_state, logits, loss, dropout_keep_prob

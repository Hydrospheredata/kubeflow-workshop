import os 
import argparse
import logging
import data_loader 
import tensorflow as tf
import numpy as np
from model import Conversation
from hbconfig import Config


def get_vocab(table_type, vocab_filename="data/tiny_processed_data/vocab"):
    if not isinstance(table_type, str) and table_type not in ("index", "string"):
        raise KeyError("table_type is not supported")

    vocab = list()
    with open("data/tiny_processed_data/vocab", "r") as file:
        for line in file.readlines():
            vocab.append(line.rstrip())
    vocab = tf.convert_to_tensor(vocab)
    
    lookup = tf.contrib.lookup
    if table_type == "index": 
        table = lookup.index_table_from_tensor(mapping=vocab, num_oov_buckets=1, default_value=1)
    elif table_type == "string":
        table = lookup.index_to_string_table_from_tensor(mapping=vocab, default_value="<unk>")
    return table


class ExportConversationWithConversion(Conversation):

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32
        self.mode = mode
        self.padding = tf.constant([[0, 0], [0, Config.data.max_seq_length]])

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self.index_table = get_vocab(table_type="index")
        self.string_table = get_vocab(table_type="string")

        self._init_placeholder(features, labels)
        self.build_graph()

        output = self._postprocess_output(self.predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            export_outputs={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(output)},
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": output})

    def _preprocess_input(self, sentence_tensor):
        words = tf.string_split(sentence_tensor[0])
        densewords = tf.sparse_tensor_to_dense(words, default_value="<pad>")
        ids = self.index_table.lookup(densewords)
        padded = tf.pad(ids, self.padding)
        sliced = tf.slice(padded, [0, 0], [1, Config.data.max_seq_length])
        return sliced 

    def _postprocess_output(self, sequence_tensor):
        words = self.string_table.lookup(tf.cast(sequence_tensor, dtype=tf.int64))
        sentence = tf.concat(words, 1)
        return tf.squeeze(sentence)

    def _init_placeholder(self, features, labels):
        self.encoder_inputs = features
        if type(features) == dict:
            self.encoder_inputs = features["input_data"]

        # transform string input to indexes
        self.encoder_inputs = self._preprocess_input(self.encoder_inputs)

        batch_size = tf.shape(self.encoder_inputs)[0]
        if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
            self.decoder_inputs = labels
            decoder_input_shift_1 = tf.slice(self.decoder_inputs, [0, 1],
                    [batch_size, Config.data.max_seq_length-1])
            pad_tokens = tf.zeros([batch_size, 1], dtype=tf.int32)

            # make target (right shift 1 from decoder_inputs)
            self.targets = tf.concat([decoder_input_shift_1, pad_tokens], axis=1)
        else:
            self.decoder_inputs = None


if __name__ == '__main__':
    # define, which configs to load
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config', help='config file name')
    args = parser.parse_args()
    tf.logging.set_verbosity(logging.INFO)

    # print Config setting
    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    # initialize configs
    Config.train.batch_size = 1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # load vocabulary
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    # create estimator
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=tf.ConfigProto(
        device_count={'GPU': 0}  # Using CPU
    ))
    conversation = ExportConversationWithConversion()
    estimator = tf.estimator.Estimator(
        model_fn=conversation.model_fn,
        model_dir=Config.train.model_dir,
        params=params, 
        config=run_config)

    # prepare serving input reciever
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {"input_data": tf.placeholder(dtype=tf.string, shape=(None, 1))})

    # export the model
    estimator.export_savedmodel(
        export_dir_base='my_model/',
        serving_input_receiver_fn=serving_input_receiver_fn)




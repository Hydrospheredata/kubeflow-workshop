#-- coding: utf-8 -*-

import argparse
import logging
import sys
import os 
import time 
import json

from hbconfig import Config
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_loader
from model import Conversation
import hook
import utils


data_dir = "/data/"
model_dir = "/models/"


def main():
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps,
    )

    tf_config = os.environ.get('TF_CONFIG', '{}')
    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
        job_name=job_name,
        task_index=task_index)

    if job_name == "ps":
        tf.logging.info("Started server!")
        server.join()

    if job_name == "worker":
        with tf.Session(server.target):
            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % task_index,
                    cluster=cluster)
                ):
                tf.logging.info("Initializing Estimator")
                conversation = Conversation()
                estimator = tf.estimator.Estimator(
                    model_fn=conversation.model_fn,
                    model_dir=Config.train.model_dir,
                    params=params,
                    config=run_config)
                        
                tf.logging.info("Initializing vocabulary")
                vocab = data_loader.load_vocab("vocab")
                Config.data.vocab_size = len(vocab)

                train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()
                train_input_fn, train_input_hook = data_loader.make_batch((train_X, train_y), batch_size=Config.model.batch_size)
                test_input_fn, test_input_hook = data_loader.make_batch((test_X, test_y), batch_size=Config.model.batch_size, scope="test")

                tf.logging.info("Initializing Specifications")
                train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
                eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)
                tf.logging.info("Run training")
                tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config', help='config file name')
    args = parser.parse_args()

    # Print Config setting
    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(" - {}: {}".format(key, value))

    main()

import logging
logging.basicConfig(level="INFO")
import tensorflow as tf
import numpy as np
import os
import argparse

from tf_collective_all_reduce.python.ops import rabit, optimizer_wrapper, broadcast_variables_hook
import rabit_starter


def model_fn(features, labels, mode, params):
    feature_columns = params.get('feature_columns')()

    logits = tf.feature_column.linear_model(
        features,
        feature_columns
    )

    average_loss = tf.losses.sigmoid_cross_entropy(
        labels,
        logits,
        reduction=tf.losses.Reduction.MEAN
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = optimizer_wrapper.DistributedOptimizer(
            tf.train.FtrlOptimizer(learning_rate=0.1)
        )
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=average_loss, train_op=train_op)


label_name = 'label'


def training_input_fn():
    def gen():
        for _ in range(100000):
            yield (2, 1, 0)
            yield (1, 2, 1)
            yield (3, 3, 0)
            yield (5, 5, 1)

    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.int64, tf.int64, tf.float32)
    )
    dataset = dataset.batch(2)
    dataset = dataset.map(
        lambda x, y, z: {'partnerid': [x], 'campaignid': [y], label_name: [z]}
    )
    dataset = dataset.map(
        lambda x: (x, x.pop(label_name))
    )
    return dataset


def test_simple_estimator():

    @rabit_starter.start(2)
    def simple_estimator(rank=None):

        estimator = tf.estimator.Estimator(
            model_fn, model_dir="model_dir",
            config=tf.estimator.RunConfig(),
            params={
                "feature_columns": lambda: [
                    tf.feature_column.categorical_column_with_hash_bucket(
                        "partnerid",
                        13,
                        dtype=tf.int64),
                    tf.feature_column.categorical_column_with_hash_bucket(
                        "campaignid",
                        3,
                        dtype=tf.int64)
                ]
            }
        )

        estimator.train(
            training_input_fn,
            steps=100,
            hooks=[broadcast_variables_hook.BroadcastGlobalVariablesHook()]
        )

    simple_estimator()

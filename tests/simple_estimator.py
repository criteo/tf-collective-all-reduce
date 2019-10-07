import tensorflow as tf
import numpy as np
import os
import argparse
from tf_collective_all_reduce import (
    DistributedOptimizer,
    BroadcastGlobalVariablesHook
)

from tf_collective_all_reduce.python.ops import rabit

tf.logging.set_verbosity(tf.logging.INFO)


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
        optimizer = DistributedOptimizer(tf.train.FtrlOptimizer(learning_rate=0.1))
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=average_loss, train_op=train_op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--rank', type=str)
    args = parser.parse_args()

    os.environ['DMLC_ROLE'] = "worker"
    os.environ['DMLC_TRACKER_URI'] = args.ip
    os.environ['DMLC_TRACKER_PORT'] = args.port
    os.environ['DMLC_RANK'] = args.rank

    print(f'Rank: {args.rank}')

    rabit.init()

    label_name = 'weighted_credited_sales_count_same_device'

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

    def gen():
        for _ in range(100000):
            yield (2, 1, 3)
            yield (1, 2, 3)
            yield (3, 3, 3)

    def training_input_fn():
        dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int64, tf.int64, tf.float32)
        )
        dataset = dataset.batch(3)
        dataset = dataset.map(
            lambda x, y, z: {'partnerid': [x], 'campaignid': [y], label_name: [z]}
        )
        dataset = dataset.map(
            lambda x: (x, x.pop(label_name))
        )
        return dataset

    estimator.train(training_input_fn, steps=10, hooks=[BroadcastGlobalVariablesHook(0)])


if __name__ == '__main__':
    main()

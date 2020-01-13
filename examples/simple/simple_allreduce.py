import tensorflow as tf
import numpy as np
import os
import argparse
import tf_collective_all_reduce
from tf_collective_all_reduce import rabit


def main():

    rabit.init()

    n_workers = rabit.get_world_size()
    print(f'World size: {n_workers}')

    rank = rabit.get_rank()
    print(f'Rank: {rank}')

    tensors_to_reduce = [
        tf.constant(13.),
        tf.constant([4.]),
        tf.constant([[1., 2., 3.]]),
        tf.constant([[4.], [5.]]),
        tf.constant([[[6., 7.], [8., 9.], [10., 11.]]]),
        tf.range(start=0, limit=500000, dtype=tf.float32)
    ]
    allreduce_res = tf_collective_all_reduce.allreduce(tensors_to_reduce)

    with tf.Session() as sess:
        allreduce_res_np = sess.run(allreduce_res)
        tensors_to_reduce_np = sess.run(tensors_to_reduce)
        [np.testing.assert_array_equal(a, b * n_workers) for a, b in
         zip(allreduce_res_np, tensors_to_reduce_np)]

    print(f"Everything's OK from {rank}!")

    rabit.finalize()


if __name__ == '__main__':
    main()
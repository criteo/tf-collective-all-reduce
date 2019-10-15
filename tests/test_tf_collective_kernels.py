import tensorflow as tf
import numpy as np

from tf_collective_all_reduce.python.ops import tf_collective_ops
from tf_collective_all_reduce.python.ops import rabit


import rabit_starter

nworkers = 2


def test_gather():
    @rabit_starter.start(nworkers)
    def gather(rank=None):
        arrays_to_gather = [
            np.array([13, 14]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[[13]]]),
            np.array([[[[1], [2]], [[3], [4]]]]),
            np.arange(500000)
        ]

        tensors_to_gather = [tf.constant(arr, dtype=tf.int32)
                            for arr in arrays_to_gather]

        gather_res = tf_collective_ops.allgather(tensors_to_gather)

        expected_res = [np.concatenate([arr for i in range(nworkers)])
                        for arr in arrays_to_gather]

        with tf.Session() as sess:
            gather_res_np = sess.run(gather_res)
            [np.testing.assert_array_equal(a, b)
            for a, b in zip(gather_res_np, expected_res)]

    gather()


def test_allreduce():
    @rabit_starter.start(nworkers)
    def allreduce(rank=None):
        arrays_to_reduce = [
            np.array([[[100.]], [[300.]]]),
            np.array([[10.], [20.], [30.], [40.]]),
            np.array([1., 2., 3.]),
            np.array([4.])
        ]

        allreduce_res = tf_collective_ops.allreduce(arrays_to_reduce)

        with tf.Session() as sess:
            allreduce_res_np = sess.run(allreduce_res)

            expected_res = [arr * nworkers for arr in arrays_to_reduce]

            [np.testing.assert_array_equal(a, b)
            for a, b in zip(expected_res, allreduce_res_np)]

    allreduce()


def test_broadcast():
    @rabit_starter.start(nworkers)
    def broadcast(rank=None):
        arrays_to_broadcast = [
            np.array([[3, 3, 3], [33, 33, 33]]),
            np.array([4, 5, 6]),
            np.array(13),
            np.array([[[1], [2]], [[3], [4]]]),
            np.arange(500000)
        ]

        broadcast_res = tf_collective_ops.broadcast(
            0,
            arrays_to_broadcast if rank == 0
            else [tf.zeros_like(tensor) for tensor in arrays_to_broadcast]
        )

        with tf.Session() as sess:
            broadcast_res_np = sess.run(broadcast_res)
            [np.testing.assert_array_equal(a, b) for a, b in
            zip(broadcast_res_np, arrays_to_broadcast)]

    broadcast()

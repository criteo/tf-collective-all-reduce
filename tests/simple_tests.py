import tensorflow as tf
import numpy as np
import os
import argparse
import tf_collective_ops as my_kernel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--rank', type=str)
    parser.add_argument('--nworkers', type=int)
    args = parser.parse_args()

    os.environ['DMLC_ROLE'] = "worker"
    os.environ['DMLC_TRACKER_URI'] = args.ip
    os.environ['DMLC_TRACKER_PORT'] = args.port
    os.environ['DMLC_RANK'] = args.rank

    print(f'Rank: {args.rank}')
    
    # AllGather tests
    
    tensors_to_gather = [
        tf.constant([13]),
        tf.constant([[1, 2, 3]]),
        tf.constant([[[13]]]),
        tf.constant([[[[1], [2]], [[3], [4]]]])
    ]

    expected_res = [
        np.array([13, 13]),
        np.array([[1, 2, 3], [1, 2, 3]]),
        np.array([[[13]], [[13]]]),
        np.array([[[[1], [2]], [[3], [4]]], [[[1], [2]], [[3], [4]]]]),
    ]

    gather_res = my_kernel.allgather(tensors_to_gather)

    with tf.Session() as sess:
        gather_res_np = sess.run(gather_res)
        print(gather_res_np)
        [np.testing.assert_array_equal(a, b) for a, b in zip(gather_res_np, expected_res)] 
    
    # Broadcast tests

    tensors_to_broadcast = [
        tf.constant([[3, 3, 3], [33, 33, 33]]),
        tf.constant([4, 5, 6]),
        tf.constant(13),
        tf.constant([[[1], [2]], [[3], [4]]])
    ]
    
    broadcast_res = my_kernel.broadcast(
        0,
        tensors_to_broadcast if args.rank == "0" \
        else [tf.zeros_like(tensor) for tensor in tensors_to_broadcast]
    )
    
    with tf.Session() as sess:
        broadcast_res_np = sess.run(broadcast_res)
        tensors_to_broadcast_np = sess.run(tensors_to_broadcast)
        print(broadcast_res_np)
        [np.testing.assert_array_equal(a, b) for a, b in zip(broadcast_res_np, tensors_to_broadcast_np)]

    # AllReduce tests

    tensors_to_reduce = [
        tf.constant([[[100.]], [[300.]]]),
        tf.constant([[10.], [20.], [30.], [40.]]),
        tf.constant([1., 2., 3.]),
        tf.constant([4.])
    ]
    allreduce_res = my_kernel.allreduce(tensors_to_reduce)

    with tf.Session() as sess:
        allreduce_res_np = sess.run(allreduce_res)
        tensors_to_reduce_np = sess.run(tensors_to_reduce)
        print(allreduce_res_np)
        [np.testing.assert_array_equal(a, b * args.nworkers) for a, b in zip(allreduce_res_np, tensors_to_reduce_np)]

    tensors_to_reduce = [
        tf.constant(13.),
        tf.constant([4.]),
        tf.constant([[1., 2., 3.]]),
        tf.constant([[4.], [5.]]),
        tf.constant([[[6., 7.], [8., 9.], [10., 11.]]])
    ]
    allreduce_res = my_kernel.allreduce(tensors_to_reduce)

    with tf.Session() as sess:
        allreduce_res_np = sess.run(allreduce_res)
        tensors_to_reduce_np = sess.run(tensors_to_reduce)
        print(allreduce_res_np)
        [np.testing.assert_array_equal(a, b * args.nworkers) for a, b in zip(allreduce_res_np, tensors_to_reduce_np)]

    print("Everything's OK!")

if __name__ == '__main__':
    main()

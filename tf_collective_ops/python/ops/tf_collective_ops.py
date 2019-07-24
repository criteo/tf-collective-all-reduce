import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


tf_collective_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_collective_ops.so'))

def broadcast(rank, tensors):
    if not isinstance(tensors, list):
        _tensors = [tensors]
    else:
        _tensors = tensors
    cast_tensors = [tf.cast(e, tf.float64) for e in _tensors]
    reduced_tensors = tf_collective_ops.broadcast(rank, tf.convert_to_tensor(len(cast_tensors), dtype=tf.uint32), cast_tensors)
    cast_reduced_tensors = [tf.cast(e, f.dtype) for e, f in zip(reduced_tensors, _tensors)]
    if not isinstance(tensors, list):
        return cast_reduced_tensors[0]
    else:
        return cast_reduced_tensors

def allreduce(tensors):
    if not isinstance(tensors, list):
        _tensors = [tensors]
    else:
        _tensors = tensors
    cast_tensors = [tf.cast(e, tf.float64) for e in _tensors]
    reduced_tensors = tf_collective_ops.allreduce(tf.convert_to_tensor(len(cast_tensors), dtype=tf.uint32), cast_tensors)
    cast_reduced_tensors = [tf.cast(e, f.dtype) for e, f in zip(reduced_tensors, _tensors)]
    if not isinstance(tensors, list):
        return cast_reduced_tensors[0]
    else:
        return cast_reduced_tensors

#allreduce = tf_collective_ops.allreduce
#broadcast = tf_collective_ops.broadcast

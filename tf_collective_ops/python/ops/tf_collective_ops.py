from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


tf_collective_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_collective_ops.so'))
allreduce = tf_collective_ops.allreduce
broadcast = tf_collective_ops.broadcast

from tf_collective_ops import broadcast
import tensorflow as tf


def broadcast_global_variables(root_rank):
    print(f"tf.global_variables(): {tf.global_variables()}")
    return broadcast_variables(tf.global_variables(), root_rank)


def broadcast_variables(variables, root_rank):
    broadcasted_vars = broadcast(root_rank, variables)
    return tf.group(*[tf.assign(var, broadcasted_var)
                      for var, broadcasted_var in zip(variables, broadcasted_vars)])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    def __init__(self, root_rank, device=''):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.device = device
        self.bcast_op = None

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = broadcast_global_variables(self.root_rank)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)
        var = tf.global_variables()
        for e in var:
            print(f"Tensor {e.name}: {session.run(e)}")

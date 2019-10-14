import tensorflow as tf
from tf_collective_all_reduce import broadcast


def broadcast_variables(variables, root_rank):
    tf.logging.info(f"Broadcasting variables: {variables}")
    broadcasted_vars = broadcast(root_rank, variables)
    return tf.group(*[tf.assign(var, broadcasted_var)
                    for var, broadcasted_var in zip(variables, broadcasted_vars)])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    def __init__(
        self,
        root_rank=0,
        device='',
        filtered_var_names=[]
    ):
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.device = device
        self.bcast_op = None
        self.filtered_var_names = filtered_var_names

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                # we want to filter out partionned variables like
                # 'embeddings:0', 'embeddings:1', 'embeddings:2', ..
                # => we test if the filter pattern ['embeddings', ..] is part of the variable
                variables = [var for var in tf.global_variables()
                             if not [var_name for var_name in self.filtered_var_names
                                     if var_name in var.name]
                            ]
                self.bcast_op = broadcast_variables(variables, self.root_rank)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)

import tensorflow as tf
from tf_collective_ops import allreduce, allgather
from tensorflow.python.training.optimizer import _deduplicate_indexed_slices


class DistributedOptimizer(tf.train.Optimizer):
    
    def __init__(self, optimizer, n_workers, name=None, use_locking=False):
        """Construct a new DistributedOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been averaged

        Args:
          optimizer:
            Optimizer to use for computing gradients and applying updates.
          name:
            Optional name prefix for the operations created when applying
            gradients. Defaults to "Distributed" followed by the provided
            optimizer type.
          use_locking:
            Whether to use locking when updating variables.
            See Optimizer.__init__ for more info.
        """
        self._optimizer = optimizer
        self.n_workers = n_workers

        def allreduce_grads(grads_vars):
            print(f"grads: {grads_vars}")
           
            grads_vars_to_gather = []
            grads_vars_to_reduce = []
            for grad_var in grads_vars:
                if isinstance(grad_var[0], tf.IndexedSlices):
                    grads_vars_to_gather.append(grad_var)
                else:
                    grads_vars_to_reduce.append(grad_var)

            print(f"grads_vars_to_gather: {grads_vars_to_gather}")
            print(f"grads_vars_to_reduce: {grads_vars_to_reduce}")

            new_grads_vars = []

            if len(grads_vars_to_gather) > 0:
                grads_to_gather, vars = zip(*grads_vars_to_gather)
                gathered_indices = allgather([grad.indices for grad in grads_to_gather])
                gathered_values = allgather([grad.values for grad in grads_to_gather])
                n_workers = tf.cast(self.n_workers, dtype=gathered_values[0].dtype)
                gathered_values = [tf.div(grad, n_workers) for grad in gathered_values]
                gathered_grads = [
                    tf.IndexedSlices(
                        indices=indices, values=values, dense_shape=grad_to_gather.dense_shape
                    ) for grad_to_gather, values, indices in zip(grads_to_gather, gathered_values, gathered_indices)
                ]
                new_grads_vars.extend(list(zip(gathered_grads, vars)))

            if len(grads_vars_to_reduce) > 0:
                grads_to_reduce, vars = zip(*grads_vars_to_reduce)
                reduced_grads = allreduce(grads_to_reduce)
                n_workers = tf.cast(self.n_workers, dtype=reduced_grads[0].dtype)
                reduced_grads = [tf.div(grad, n_workers) for grad in reduced_grads]
                new_grads_vars.extend(list(zip(reduced_grads, vars)))

            return new_grads_vars

        self._allreduce_grads = allreduce_grads

        super(DistributedOptimizer, self).__init__(
            name="Myoptimizer", use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if self.n_workers > 1:
            avg_grads = self._allreduce_grads(gradients)
            return avg_grads
        else:
            return gradients

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)

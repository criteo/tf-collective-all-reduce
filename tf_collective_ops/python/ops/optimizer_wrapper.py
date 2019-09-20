import tensorflow as tf
from tf_collective_ops import allreduce, allgather


class DistributedOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, n_workers, name=None, use_locking=False, average=True):
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
        self.average = average
        self.dependencies = []

        super(DistributedOptimizer, self).__init__(
            name="DistributedOptimizer", use_locking=use_locking)

    def _op_with_dependencies(self, op, grad):
        with tf.control_dependencies(self.dependencies):
            res = op(grad)
        self.dependencies += res if isinstance(res, list) else [res]
        return res

    def _allgather(self, grad):
        return self._op_with_dependencies(allgather, grad)

    def _allreduce(self, grad):
        return self._op_with_dependencies(allreduce, grad)

    def _allreduce_grad(self, tensor):
        n_workers = tf.cast(self.n_workers, dtype=tensor.dtype)
        if isinstance(tensor, tf.IndexedSlices):
            values = self._allgather([tensor.values])[0]
            indices = self._allgather([tensor.indices])[0]
            if self.average:
                values = tf.div(values, n_workers)
            return tf.IndexedSlices(
                indices=indices,
                values=values,
                dense_shape=tensor.dense_shape)
        else:
            summed_tensor = self._allreduce([tensor])[0]
            if self.average:
                summed_tensor = tf.div(summed_tensor, n_workers)
            return summed_tensor

    def _allreduce_grads(self, grads):
        tf.logging.info("Creating allreduce grad ops")
        return [self._allreduce_grad(grad) for grad in grads]

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        if self.n_workers > 1:
            grads, vars = zip(*gradients)
            agg_grads = self._allreduce_grads(grads)
            return list(zip(agg_grads, vars))
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

import tensorflow as tf
from tf_collective_ops import allreduce as myallreduce


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

        def allreduce_grad(grad):
            n_workers = tf.cast(self.n_workers, dtype=grad.dtype)
            summed_grad = myallreduce(grad)
            new_grad = tf.div(summed_grad, n_workers)
            return new_grad

        def allreduce_grads(grads):
            grads = [
                tf.convert_to_tensor(grad) if grad is not None and isinstance(grad, tf.IndexedSlices) 
                else grad for grad in grads
            ]

            # return [allreduce_grad(grads[1])]
            return [allreduce_grad(grad) if grad is not None else grad for grad in grads]

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
            grads, vars = zip(*gradients)
            avg_grads = self._allreduce_grads(grads)
            return list(zip(avg_grads, vars))
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

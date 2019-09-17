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

        def allreduce_grads(grads_vars):
            def op_with_dependencies(dependencies, op, grads):
                with tf.control_dependencies(dependencies):
                    res = op(grads)
                dependencies += res if isinstance(res, list) else list(res)
                return res

            def _allgather(dependencies, grads):
                return op_with_dependencies(dependencies, allgather, grads)

            def _allreduce(dependencies, grads):
                return op_with_dependencies(dependencies, allreduce, grads)

            dependencies = []
            grads_vars_to_gather = []
            grads_vars_to_reduce = []
            for grad, var in grads_vars:
                if isinstance(grad, tf.IndexedSlices):
                    grads_vars_to_gather.append((grad, var))
                else:
                    grads_vars_to_reduce.append((grad, var))

            new_grads_vars = []

            if len(grads_vars_to_gather) > 0:
                grads, _ = zip(*grads_vars_to_gather)
                gathered_indices = _allgather(dependencies, [grad.indices for grad in grads])
                gathered_values = _allgather(dependencies, [grad.values for grad in grads])
                if self.average:
                    gathered_values = [tf.div(gathered_value, self.n_workers) for gathered_value in gathered_values]
                new_grads_vars.extend([
                    (tf.IndexedSlices(
                        indices=indices, values=values, dense_shape=grad.dense_shape
                    ), var) for (grad, var), values, indices in zip(grads_vars_to_gather, gathered_values, gathered_indices)
                ])

            if len(grads_vars_to_reduce) > 0:
                grads, _ = zip(*grads_vars_to_reduce)
                reduced_grads = _allreduce(dependencies, grads)
                if self.average:
                    reduced_grads = [tf.div(reduced_grad, self.n_workers) for reduced_grad in reduced_grads]
                new_grads_vars.extend([(reduced_grad, var) for (_, var), reduced_grad in zip(grads_vars_to_reduce, reduced_grads)])

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
        grads_vars = self._optimizer.compute_gradients(*args, **kwargs)
        if self.n_workers > 1:
            return self._allreduce_grads(grads_vars)
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

import tensorflow as tf
from tf_collective_all_reduce import allreduce, allgather
from collections import defaultdict


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
          average:
            Whether to compute the average of reduced gradients.
            If the loss to optimize is an average, this parameter must be set to True
            Otherwise, it is most likely that it should be set to False
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

    def _allreduce_grads(self, grads_vars):
        tf.logging.info(f"Creating allreduce ops for {grads_vars}")
        grads_vars_to_gather = defaultdict(list)
        grads_vars_to_reduce = defaultdict(list)
        for grad, var in grads_vars:
            if isinstance(grad, tf.IndexedSlices):
                grads_vars_to_gather[(grad.dtype, grad.indices.dtype)].append((grad, var))
            else:
                grads_vars_to_reduce[grad.dtype].append((grad, var))

        new_grads_vars = []

        if len(grads_vars_to_gather) > 0:
            for grads_vars in grads_vars_to_gather.values():
                grads, _ = zip(*grads_vars)
                gathered_indices = self._allgather([grad.indices for grad in grads])
                gathered_values = self._allgather([grad.values for grad in grads])
                if self.average:
                    gathered_values = [tf.div(gathered_value, self.n_workers)
                                       for gathered_value in gathered_values]
                new_grads_vars.extend([
                    (
                        tf.IndexedSlices(
                            indices=indices, values=values, dense_shape=grad.dense_shape
                        ),
                        var
                    )
                    for (grad, var), values, indices
                    in zip(grads_vars, gathered_values, gathered_indices)
                ])

        if len(grads_vars_to_reduce) > 0:
            for grads_vars in grads_vars_to_reduce.values():
                grads, _ = zip(*grads_vars)
                reduced_grads = self._allreduce(grads)
                if self.average:
                    reduced_grads = [
                        tf.div(reduced_grad, self.n_workers) for reduced_grad in reduced_grads
                    ]
                new_grads_vars.extend([
                    (reduced_grad, var) for (_, var), reduced_grad in zip(grads_vars, reduced_grads)
                ])

        return new_grads_vars

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
            return grads_vars

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

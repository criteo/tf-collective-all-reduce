
import tensorflow as tf

from collections import defaultdict
from tensorflow.python.training import optimizer as opt

from tf_collective_all_reduce import allreduce, allgather
from tf_collective_all_reduce.python.ops.compression import Compression


class DistributedOptimizer(tf.train.Optimizer):

    def __init__(
            self, optimizer, n_workers, name=None,
            use_locking=False, average=True,
            indices_compression=Compression.none,
            values_compression=Compression.none,
            group_gradients=True
    ):
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
          indices_compression:
            Compression algorithm to apply on IndexedSlices indices.
            Indices of IndexedSlices are mostly int64. Most of the time, the range
            of indices can be coded with less than 64 bits.
          values_compression:
            Compression algorithm to apply on gradient values
          group_gradients:
            Default: True
            Whether to send all gradients in a group to allreduce/allgather
            faster but consumes more memory
        """
        self._optimizer = optimizer
        self.n_workers = n_workers
        self.average = average
        self.indices_compression = indices_compression
        self.values_compression = values_compression
        self.group_gradients = group_gradients
        self.dependencies = []

        super(DistributedOptimizer, self).__init__(
            name="DistributedOptimizer", use_locking=use_locking)

    def _op_with_dependencies(self, op, tensors, compression=Compression.none):
        with tf.control_dependencies(self.dependencies):
            compressed_tensors, dtypes = zip(*[compression.compress(tensor) for tensor in tensors])
            results = op(compressed_tensors)
            decompressed_tensors = [
                compression.decompress(result, dtype) for result, dtype in zip(results, dtypes)
            ]
        self.dependencies += \
            decompressed_tensors if isinstance(decompressed_tensors, list) \
            else [decompressed_tensors]
        return decompressed_tensors

    def _allgather(self, tensors, compression=Compression.none):
        return self._op_with_dependencies(allgather, tensors, compression)

    def _allreduce(self, tensors, compression=Compression.none):
        return self._op_with_dependencies(allreduce, tensors, compression)

    def _create_deduplicated_indexed_slices(self, indices, values, dense_shape):
        values, indices = opt._deduplicate_indexed_slices(values, indices)
        return tf.IndexedSlices(
            indices=indices,
            values=values,
            dense_shape=dense_shape
        )

    def _allreduce_grads_group(self, grads_vars):
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
                grads = [self._create_deduplicated_indexed_slices(
                            grad.indices,
                            grad.values,
                            grad.dense_shape)
                         for grad in grads]
                gathered_indices = \
                    self._allgather([grad.indices for grad in grads], self.indices_compression)
                gathered_values = \
                    self._allgather([grad.values for grad in grads], self.values_compression)
                if self.average:
                    gathered_values = [tf.div(value, self.n_workers)
                                       for value in gathered_values]
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
                grads, vars = zip(*grads_vars)
                reduced_grads = self._allreduce(grads, self.values_compression)
                if self.average:
                    reduced_grads = [
                        tf.div(grad, self.n_workers) for grad in reduced_grads
                    ]
                new_grads_vars.extend(zip(reduced_grads, vars))

        return new_grads_vars

    def _allreduce_grad_one_by_one(self, tensor):
        n_workers = tf.cast(self.n_workers, dtype=tensor.dtype)
        if isinstance(tensor, tf.IndexedSlices):
            dedup_tensor = self._create_deduplicated_indexed_slices(
                tensor.indices,
                tensor.values,
                tensor.dense_shape)
            indices = self._allgather([dedup_tensor.indices], self.values_compression)[0]
            values = self._allgather([dedup_tensor.values], self.indices_compression)[0]
            if self.average:
                values = tf.div(values, n_workers)
            return tf.IndexedSlices(
                indices=indices,
                values=values,
                dense_shape=tensor.dense_shape)
        else:
            summed_tensor = self._allreduce([tensor], self.values_compression)[0]
            if self.average:
                summed_tensor = tf.div(summed_tensor, n_workers)
            return summed_tensor

    def _allreduce_grads_one_by_one(self, grads_vars):
        grads, vars = zip(*grads_vars)
        aggregated_grads = [self._allreduce_grad_one_by_one(grad) for grad in grads]
        return list(zip(aggregated_grads, vars))

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        grads_vars = self._optimizer.compute_gradients(*args, **kwargs)
        if self.n_workers > 1:
            tf.logging.info(f"Creating allreduce ops for {grads_vars}")
            if self.group_gradients:
                return self._allreduce_grads_group(grads_vars)
            else:
                return self._allreduce_grads_one_by_one(grads_vars)
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

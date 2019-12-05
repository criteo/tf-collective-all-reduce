import logging
import tensorflow as tf
from tensorflow.python.estimator.training import _EvalStatus
from tensorflow.python import ops
import os
import skein
import time
from threading import Thread
from tf_yarn.tasks import logging as tf_yarn_logging
tf_yarn_logging.setup()

from tf_yarn import _task_commons, cluster, event
from tf_yarn.tasks.evaluator_fn import evaluator_fn
from tf_collective_all_reduce.python.tf_yarn import tracker
from tf_collective_all_reduce.python.ops import rabit

logger = logging.getLogger(__name__)


def _setup_tracker(client):
    host_port = event.wait(client, "chief:0/tracker")
    tf.logging.info(f"Got tracker url {host_port}")
    host, port = host_port.split(":")
    os.environ['DMLC_TRACKER_URI'] = host
    os.environ['DMLC_TRACKER_PORT'] = port


def _start_tracker(client, n_workers: int):
    tf.logging.info(f"Starting tracker with {n_workers} workers")
    rabit_context = tracker.RabitTracker(
        hostIP=tracker.get_host_ip(),
        nslave=n_workers,
        # will do bind(0) -> choose a random port
        port=0,
        port_end=1)
    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join, daemon=True)
    thread.start()

    event.broadcast(
        client,
        f"{cluster.get_task()}/tracker",
        f"{rabit_context.hostIP}:{rabit_context.port}"
    )
    return thread


def _worker_fn(task_type, task_id, client):
    os.environ['DMLC_RANK'] = "0" if task_type == 'chief' else f"{task_id + 1}"
    os.environ['DMLC_ROLE'] = "worker"

    cluster_tasks = _task_commons._get_cluster_tasks(client)

    logger.info(cluster_tasks)

    if task_type == 'chief':
        _start_tracker(client, len(cluster_tasks))

    _setup_tracker(client)

    rabit.init()

    experiment = _task_commons._get_experiment(client)

    if task_type != 'chief':
        # Overwrite config to do nothing but training to improve training speed
        experiment.estimator._model_dir = "."
        new_config = experiment.estimator.config.replace(
            save_summary_steps=None,
            save_checkpoints_steps=None,
            save_checkpoints_secs=None,
            log_step_count_steps=None
        )
        experiment.estimator._config = new_config

    logger.info(f"start training..")

    experiment.estimator.train(
        experiment.train_spec.input_fn,
        hooks=experiment.train_spec.hooks,
        max_steps=experiment.train_spec.max_steps)


def main():
    client = skein.ApplicationClient.from_current()
    task_type, task_id = cluster.get_task_description()
    task = cluster.get_task()
    event.init_event(client, task, f"127.0.0.1:0")
    _task_commons._setup_container_logs(client)

    if task_type in ['chief', 'worker']:
        _worker_fn(task_type, task_id, client)
    elif task_type == 'evaluator':
        evaluator_fn(client)
    else:
        logger.error(f'Unknown task type {task_type}')

    event.stop_event(client, task, None)


if __name__ == "__main__":
    main()

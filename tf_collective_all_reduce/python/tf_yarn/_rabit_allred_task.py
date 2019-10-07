import logging
import tensorflow as tf
import os
import skein
from threading import Thread

from tf_yarn import _task_commons, cluster, event
from tf_collective_all_reduce.python.tf_yarn import tracker

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
        nslave=n_workers)
    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join, daemon=True)
    thread.start()

    event.broadcast(
        client,
        f"{cluster.get_task()}/tracker",
        f"{rabit_context.hostIP}:{rabit_context.port}"
    )
    return thread


def main():
    client = skein.ApplicationClient.from_current()
    task_type, task_id = cluster.get_task_description()
    task = cluster.get_task()
    event.init_event(client, task, f"127.0.0.1:0")
    _task_commons._setup_container_logs(client)

    os.environ['DMLC_RANK'] = "0" if task_type == 'chief' else f"{task_id + 1}"
    os.environ['DMLC_ROLE'] = "worker"

    cluster_tasks = _task_commons._get_cluster_tasks(client)

    logger.info(cluster_tasks)

    if task_type == 'chief':
        _start_tracker(client, len(cluster_tasks))

    _setup_tracker(client)

    experiment = _task_commons._get_experiment(client)

    if task_type != 'chief':
        experiment.estimator._model_dir = "."

    logger.info(f"start training..")

    experiment.estimator.train(
        experiment.train_spec.input_fn,
        hooks=experiment.train_spec.hooks,
        max_steps=experiment.train_spec.max_steps)

    event.stop_event(client, task, None)


if __name__ == "__main__":
    _task_commons._process_arguments()
    main()

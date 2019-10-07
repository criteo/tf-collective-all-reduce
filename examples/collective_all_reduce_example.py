"""
Example of custom distributed training with one node trainings
and using https://gitlab.criteois.com/g.racic/rabit-fork AllReduce
"""
import logging
import os
import pwd
import getpass
from subprocess import check_output
import skein
import tensorflow as tf
import winequality
from datetime import datetime

from tf_yarn import Experiment, TaskSpec, packaging, run_on_yarn

from tf_collective_all_reduce import (
    DistributedOptimizer,
    BroadcastGlobalVariablesHook
)

logging.basicConfig(level="INFO")

USER = getpass.getuser()

"""
1. Download winequality-*.csv from the Wine Quality dataset at UCI
   ML repository
   (https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Upload it to HDFS
3. Pass a full URI to either of the CSV files to the example
"""
WINE_EQUALITY_FILE = f"{packaging.get_default_fs()}/user/{USER}/tf_yarn_test/winequality-red.csv"

"""
Output path of the learned model on hdfs
"""
HDFS_DIR = (f"{packaging.get_default_fs()}/user/{USER}"
            f"/tf_yarn_test/tf_yarn_{int(datetime.now().timestamp())}")

NB_WORKERS = 1


def experiment_fn() -> Experiment:
    def train_input_fn():
        train_data, test_data = winequality.get_train_eval_datasets(WINE_EQUALITY_FILE)
        return (train_data.shuffle(1000)
                .batch(128)
                .repeat())

    estimator = tf.estimator.LinearClassifier(
        optimizer=DistributedOptimizer(tf.train.FtrlOptimizer(learning_rate=0.1)),
        feature_columns=winequality.get_feature_columns(),
        model_dir=f"{HDFS_DIR}",
        n_classes=winequality.get_n_classes())

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=1000,
        hooks=[BroadcastGlobalVariablesHook(0)]
    )
    return Experiment(estimator, train_spec, tf.estimator.EvalSpec(lambda: True))


if __name__ == "__main__":

    pyenv_zip_path, env_name = packaging.upload_env_to_hdfs()
    editable_requirements = packaging.get_editable_requirements_from_current_venv()

    with skein.Client() as client:
        run_on_yarn(
            pyenv_zip_path,
            experiment_fn,
            task_specs={
                "chief": TaskSpec(memory="1 GiB", vcores=1),
                "worker": TaskSpec(memory="1 GiB", vcores=1, instances=NB_WORKERS)
            },
            files={
                **editable_requirements,
                os.path.basename(winequality.__file__): winequality.__file__,
            },
            skein_client=client,
            custom_task_module="tf_collective_all_reduce.python.tf_yarn._rabit_allred_task"
        )

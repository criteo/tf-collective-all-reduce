import logging
import multiprocessing
import os
import sys

from threading import Thread
from functools import wraps

from tf_collective_all_reduce.python.ops import rabit
from tf_collective_all_reduce.python.tf_yarn import tracker

_logger = logging.getLogger(__name__)


def start(nworkers):
    def rabit_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ip = "127.0.0.1"
            port = 9095

            def _single_process(rank):
                os.environ['DMLC_ROLE'] = "worker"
                os.environ['DMLC_TRACKER_URI'] = ip
                os.environ['DMLC_TRACKER_PORT'] = str(port)
                os.environ['DMLC_RANK'] = str(rank)
                _logger.info(f'Rank: {rank}')

                rabit.init()

                func(rank=rank)

            def _start_tracker():
                try:
                    _logger.info("start tracker")
                    rabit_context = tracker.RabitTracker(
                        hostIP=ip,
                        nslave=nworkers,
                        port=port)
                    rabit_context.start(nworkers)
                    return rabit_context
                except Exception:
                    print(sys.exc_info())

            rabit_context = _start_tracker()

            _logger.info(f"rabit context alive={rabit_context.alive()}"
                         f" port={rabit_context.port} ip={rabit_context.hostIP}")

            process = []
            for i in range(nworkers):
                p = multiprocessing.Process(target=_single_process, args=(i,))
                p.start()
                process.append(p)

            for p in process:
                p.join()
                assert not p.exitcode

        return wrapper
    return rabit_decorator

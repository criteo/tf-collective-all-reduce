# tf-collective-all-reduce

Lightweight framework for distributing machine learning training based on [Rabit](https://github.com/dmlc/rabit) for the communication layer. We borrowed [Horovod's](https://github.com/horovod/horovod) concepts for the TensorFlow optimizer wrapper.

## Installation

```
git clone https://github.com/criteo/tf-collective-all-reduce
python3.6 -m venv tf_env
. tf_env/bin/activate
pip install tensorflow==1.12.2
pushd tf-collective-all-reduce
  ./install.sh
  pip install -e .
popd
```

## Prerequisites

tf-collective-all-reduce only supports Python ≥3.6


## Run tests

```
pip install -r tests-requirements.txt
pytest -s
```

## Local run with [dmlc-submit](https://github.com/dmlc/dmlc-core/tree/master/tracker)

```
./libs/dmlc-core/tracker/dmlc-submit --cluster local --num-workers 2 python examples/simple/simple_allreduce.py
```

## Run on a Hadoop cluster with [tf-yarn](https://github.com/criteo/tf-yarn)

Run [collective_all_reduce_example](https://github.com/criteo/tf-collective-all-reduce/blob/master/examples/collective_all_reduce_example.py)

```
cd examples/tf-yarn
python collective_all_reduce_example.py
```


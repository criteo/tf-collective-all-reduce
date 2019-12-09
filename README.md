# tf-collective-all-reduce

Lightweight framework for distributing machine learning training based on [Rabit](https://github.com/dmlc/rabit) for the communication layer. We borrowed [Horovod](https://github.com/horovod/horovod)'s concepts for the TensorFlow optimizer wrapper.

# Installation

```
python3.6 -m venv tf_env
. tf_env/bin/activate
pip install tensorflow==1.12.2
./install.sh
pip install -e .
```


# ReBuild

Compiles sources and executes example below as healthcheck

```
./rebuild.sh

```

# Run locally

```
pytest tests/test_simple_estimator.py
```

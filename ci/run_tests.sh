#!/bin/bash

pushd /root/tf-collective-all-reduce
  python3.6 -m venv /root/tf_env
  . /root/tf_env/bin/activate
  pip install tensorflow==1.12.2
  ./install.sh
  pip install -e .
  pip install -r tests-requirements.txt
  pytest -s tests
 popd

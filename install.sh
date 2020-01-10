#!/bin/bash

set -x

pushd ..

  rm -rf dmlc-core
  rm -rf rabit

  git clone https://github.com/dmlc/dmlc-core
  pushd dmlc-core
    make
  popd

  git clone https://github.com/criteo-forks/rabit
  pushd rabit
    make
  popd

popd

make clean collective_ops

set +x

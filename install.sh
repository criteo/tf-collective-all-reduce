#!/bin/bash

set -x

rm -rf libs

mkdir libs

pushd libs

  curl -L "https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.gz" -o boost_1_70_0.tar.gz
  tar -xzf  boost_1_70_0.tar.gz
  rm boost_1_70_0.tar.gz

  git clone git@gitlab.criteois.com:g.racic/dmlc-core.git
  pushd dmlc-core
    make
  popd

  git clone git@github.com:criteo-forks/rabit.git
  pushd rabit
    make
  popd

popd

make clean collective_ops

set +x

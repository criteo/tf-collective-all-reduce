

mkdir libs

pushd libs

  curl -L "https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.gz" -o boost_1_70_0.tar.gz
  tar -xzvf  boost_1_70_0.tar.gz
  rm boost_1_70_0.tar.gz

  git clone git@gitlab.criteois.com:g.racic/dmlc-core.git
  pushd dmlc-core
    make
  popd

  git clone git@gitlab.criteois.com:g.racic/rabit-fork.git rabit
  pushd rabit
    make
  popd

popd

python3.6 -m venv tf_env
. tf_env/bin/activate
pip install tensorflow==1.12.2

make collective_ops

pip install -e .

deactivate

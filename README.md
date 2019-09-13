# Installation

```
./install.sh

```


# Run locally

Setup dmlc master

```
$./libs/dmlc-core/tracker/dmlc-submit --cluster=tf-yarn --num-workers=2 --host-ip 127.0.0.1 a
```

Run both collective-all-reduce scripts (will register to rabit server and execute when registered)

```
. libs/tf_env/bin/activate

python tests/simple_tests.py --ip 127.0.0.1 --port 9091 --rank 0 --nworkers 2
```

```
. libs/tf_env/bin/activate

python tests/simple_tests.py --ip 127.0.0.1 --port 9091 --rank 1 --nworkers 2
```


make clean
make pip_pkg

./libs/dmlc-core/tracker/dmlc-submit --cluster=tf-yarn --num-workers=2 --host-ip 127.0.0.1 a > tracker.log 2>&1 &
TRACKER_PID=$!

python tests/simple_tests.py --ip 127.0.0.1 --port 9091 --rank 0 --nworkers 2 > worker_0.log 2>&1 & 

python tests/simple_tests.py --ip 127.0.0.1 --port 9091 --rank 1 --nworkers 2

kill $TRACKER_PID

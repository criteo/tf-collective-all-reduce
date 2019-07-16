CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

COLLECTIVE_OPS_SRC = $(wildcard tf_collective_ops/cc/kernels/*.cc) $(wildcard tf_collective_ops/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11 -c -v
LDFLAGS = -shared ${TF_LFLAGS}
RABIT_OBJS = ../rabit/c_api.o ../rabit/allreduce_base.o ../rabit/allreduce_robust.o ../rabit/engine.o
RABIT_PATH = -I/home/g.racic

COLLECTIVE_OPS_LIB = tf_collective_ops/python/ops/_collective_ops.so

collective_ops: $(COLLECTIVE_OPS_LIB)

COLLECTIVE_OPS_OBJS = tf_collective_kernels.o tf_collective_ops.o

$(COLLECTIVE_OPS_OBJS): $(COLLECTIVE_OPS_SRC)
	$(CXX) $(CFLAGS) $^ $(RABIT_PATH)

$(COLLECTIVE_OPS_LIB): $(COLLECTIVE_OPS_OBJS)
	$(CXX) -v -o $@ $(LDFLAGS) $(RABIT_OBJS) $^

pip_pkg: $(COLLECTIVE_OPS_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(COLLECTIVE_OPS_LIB) $(COLLECTIVE_OPS_OBJS) 

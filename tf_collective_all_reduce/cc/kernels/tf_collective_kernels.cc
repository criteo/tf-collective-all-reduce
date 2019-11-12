#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include <chrono>

#include <numeric>

#include "rabit/include/rabit/c_api.h"

#define REGISTER_KERNEL_ALLREDUCE(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Allreduce").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AllreduceOp<type>)

#define REGISTER_KERNEL_BROADCAST(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Broadcast").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BroadcastOp<type>)

#define REGISTER_KERNEL_ALLGATHER(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Allgather").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AllgatherOp<type>)

#define ALLREDUCE_SUM 2

#define RABIT_ENUM_SIZE_T 5

#define LOG_LEVEL WARNING

#define _LOG(lvl, str) \
    if (lvl >= LOG_LEVEL) \
        LOG(lvl) << str

using namespace std::chrono;
using namespace tensorflow;

namespace thx {

template <typename T>
int getTypeId() {
    static int typeId = -1;
    if (typeId < 0) {
        if (std::is_same<T, int8>::value)
            typeId = 0;
        if (std::is_same<T, uint8>::value)
            typeId = 1;
        if (std::is_same<T, int32>::value)
            typeId = 2;
        if (std::is_same<T, uint32>::value)
            typeId = 3;
        if (std::is_same<T, int64>::value)
            typeId = 4;
        if (std::is_same<T, uint64>::value)
            typeId = 5;
        if (std::is_same<T, float>::value)
            typeId = 6;
        if (std::is_same<T, double>::value)
            typeId = 7;
    }
    return typeId;
}

template <typename T>
class AllreduceOp : public OpKernel {
public:
  explicit AllreduceOp(OpKernelConstruction* context)
      : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    _LOG(INFO, std::string("Entering in allreduce op"));
    const int n_tensors = context->input(0).scalar<int>()();
    for (int i = 1; i <= n_tensors; i++) {
        auto start = high_resolution_clock::now(); 
        const Tensor& input_tensor = context->input(i);
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(i-1, input_tensor.shape(),
                                                     &output_tensor));

        auto input_flat = input_tensor.flat<T>();
        auto output_flat = output_tensor->flat<T>();

        const int N = input_flat.size();
        for (int i = 0; i < N; i++) {
            output_flat(i) = input_flat(i);
        }

        RabitAllreduce(
            (void *)(output_flat.data()), output_flat.size(),
            getTypeId<T>(), ALLREDUCE_SUM, nullptr, nullptr
        );
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start);
        _LOG(INFO, std::string("Time taken by function: ") << duration.count() << " microseconds");
    }
    _LOG(INFO, std::string("Exiting allreduce op"));
  }
};

REGISTER_KERNEL_ALLREDUCE(int8);
REGISTER_KERNEL_ALLREDUCE(uint8);
REGISTER_KERNEL_ALLREDUCE(int32);
REGISTER_KERNEL_ALLREDUCE(uint32);
REGISTER_KERNEL_ALLREDUCE(int64);
REGISTER_KERNEL_ALLREDUCE(uint64);
REGISTER_KERNEL_ALLREDUCE(float);
REGISTER_KERNEL_ALLREDUCE(double);

template <typename T>
class BroadcastOp : public OpKernel {
public:
  explicit BroadcastOp(OpKernelConstruction* context)
      : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    _LOG(INFO, std::string("Entering in broadcast op"));
    auto cur_rank = RabitGetRank();
    auto sender_rank = context->input(0).scalar<int>()();
    auto n_tensors = context->input(1).scalar<int>()();
    for (int i = 0; i < n_tensors; i++) {
        auto input_tensor = context->input(i+2);
        Tensor* output_tensor = nullptr;

        auto input_flat = input_tensor.flat<T>();
        unsigned long input_total_size = input_flat.size() * sizeof(T);
        RabitBroadcast((void *)(&input_total_size), sizeof(input_total_size), sender_rank);

        if (cur_rank == sender_rank) {
            context->set_output(i, input_tensor);
            RabitBroadcast((void *)(input_flat.data()), input_total_size, sender_rank);
        }
        else {
            OP_REQUIRES_OK(
            context, context->allocate_output(i, input_tensor.shape(), &output_tensor));
            auto output_flat = output_tensor->flat<T>();
            RabitBroadcast((void *)(output_flat.data()), input_total_size, sender_rank);
        }
    }
    _LOG(INFO, std::string("Exiting broadcast op"));
  }
};

REGISTER_KERNEL_BROADCAST(int8);
REGISTER_KERNEL_BROADCAST(uint8);
REGISTER_KERNEL_BROADCAST(int32);
REGISTER_KERNEL_BROADCAST(uint32);
REGISTER_KERNEL_BROADCAST(int64);
REGISTER_KERNEL_BROADCAST(uint64);
REGISTER_KERNEL_BROADCAST(float);
REGISTER_KERNEL_BROADCAST(double);

template <typename T>
class AllgatherOp : public OpKernel {
public:
  explicit AllgatherOp(OpKernelConstruction* context)
      : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
      _LOG(INFO, std::string("Entering in allgather op"));
      const int n_tensors = context->input(0).scalar<int>()();
      for (int i = 1; i <= n_tensors; i++) {
        auto start = high_resolution_clock::now();
        const Tensor& input_tensor = context->input(i);
        auto input_shape = input_tensor.shape();
        auto input_flat = input_tensor.flat<T>();

        Tensor* output_tensor = NULL;

        int n_workers = RabitGetWorldSize();
        int rank = RabitGetRank();
        int ringPrevRank = RabitGetRingPrevRank();
        size_t n_slicesPerWorker[n_workers];
        n_slicesPerWorker[rank] = input_shape.dim_size(0);
        RabitAllgather((void*)(n_slicesPerWorker), n_workers, rank, 1, 1, RABIT_ENUM_SIZE_T);

        size_t n_slicesBefore = std::accumulate(n_slicesPerWorker, n_slicesPerWorker + rank, 0, std::plus<size_t>());
        size_t n_slices = n_slicesBefore + std::accumulate(n_slicesPerWorker + rank, n_slicesPerWorker + n_workers, 0, std::plus<size_t>());

        auto output_shape = TensorShape();
        output_shape.AppendShape(input_shape);
        output_shape.set_dim(0, n_slices);
        OP_REQUIRES_OK(context, context->allocate_output(i-1, output_shape,
                                                     &output_tensor));

        size_t n_elemsPerSlice = input_flat.size() / n_slicesPerWorker[rank];
        auto output_flat = output_tensor->flat<T>();
        size_t beginIndex = n_slicesBefore * n_elemsPerSlice;
        for (int j = 0; j < input_flat.size(); j++) {
            output_flat(beginIndex + j) = input_flat(j);
        }

        RabitAllgather((void *)output_flat.data(), n_elemsPerSlice * n_slices, beginIndex, input_flat.size(),
                n_slicesPerWorker[ringPrevRank] * n_elemsPerSlice, getTypeId<T>());
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        _LOG(INFO, std::string("Time taken by function: ") << duration.count() << " microseconds");
     }
     _LOG(INFO, std::string("Exiting allgather op"));
  }
};

REGISTER_KERNEL_ALLGATHER(int8);
REGISTER_KERNEL_ALLGATHER(uint8);
REGISTER_KERNEL_ALLGATHER(int32);
REGISTER_KERNEL_ALLGATHER(uint32);
REGISTER_KERNEL_ALLGATHER(int64);
REGISTER_KERNEL_ALLGATHER(uint64);
REGISTER_KERNEL_ALLGATHER(float);
REGISTER_KERNEL_ALLGATHER(double);

}

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/integral_types.h"

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

void initRabit() {
    static bool isInitialized = false;
    if (!isInitialized) {
        LOG(INFO) << "Initializing Rabit";
        RabitInit(0, nullptr);
        LOG(INFO) << "Rabit is initialized";
        LOG(INFO) << "Rank: " << RabitGetRank();
        isInitialized = true;
    }
}

template <typename T>
class AllreduceOp : public OpKernel {
public:
  explicit AllreduceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    initRabit();
  }

  void Compute(OpKernelContext* context) override {
    const int n_tensors = context->input(0).scalar<int>()();
    //LOG(INFO) << "n_tensors: " << n_tensors;
    for (int i = 1; i <= n_tensors; i++) {
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

        //LOG(INFO) << "Before allreduce (output_flat): " << output_flat;
        RabitAllreduce(
            (void *)(output_flat.data()), output_flat.size(),
            getTypeId<T>(), ALLREDUCE_SUM, nullptr, nullptr
        );
        //LOG(INFO) << "After allreduce: " << output_flat;
    }
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
    initRabit();    
  }

  void Compute(OpKernelContext* context) override {
    auto cur_rank = RabitGetRank();
    auto sender_rank = context->input(0).scalar<int>()();
    auto n_tensors = context->input(1).scalar<int>()();
    //LOG(INFO) << "n_tensors: " << n_tensors;
    for (int i = 0; i < n_tensors; i++) {
        auto input_tensor = context->input(i+2);
        Tensor* output_tensor = nullptr;

        auto input_flat = input_tensor.flat<T>();
        unsigned long input_total_size = input_flat.size() * sizeof(T);
        RabitBroadcast((void *)(&input_total_size), sizeof(input_total_size), sender_rank);

        if (cur_rank == sender_rank) {
            context->set_output(i, input_tensor);
            RabitBroadcast((void *)(input_flat.data()), input_total_size, sender_rank);
            //LOG(INFO) << "Broadcasted (sender): " << input_flat << std::endl;
        }
        else {
            OP_REQUIRES_OK(
            context, context->allocate_output(i, input_tensor.shape(), &output_tensor));
            auto output_flat = output_tensor->flat<T>();
            RabitBroadcast((void *)(output_flat.data()), input_total_size, sender_rank);
            //LOG(INFO) << "Received: " << output_flat;
        }
    }
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
    initRabit();
  }

  void Compute(OpKernelContext* context) override {
      const int n_tensors = context->input(0).scalar<int>()();
      for (int i = 1; i <= n_tensors; i++) {  
        const Tensor& input_tensor = context->input(i);
        auto input_shape = input_tensor.shape();
        auto input_flat = input_tensor.flat<T>();

        Tensor* output_tensor = NULL;
        
        uint32 n_d0 = input_shape.dim_size(0);
        size_t nelems_per_slice = input_flat.size() / n_d0;
        
        RabitAllreduce((void *)(&n_d0), 1, getTypeId<uint32>(), ALLREDUCE_SUM, nullptr, nullptr);

        auto output_shape = TensorShape();
        output_shape.AppendShape(input_shape);
        output_shape.set_dim(0, n_d0);
        OP_REQUIRES_OK(context, context->allocate_output(i-1, output_shape,
                                                     &output_tensor));

        auto output_flat = output_tensor->flat<T>();

        for (int j = 0; j < input_flat.size(); j++) {
            output_flat(j) = input_flat(j);
        }

        RabitAllgather((void *)output_flat.data(), input_flat.size(), getTypeId<T>(), nelems_per_slice * n_d0);
    }
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

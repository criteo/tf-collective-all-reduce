#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/integral_types.h"


#include "rabit/include/rabit/c_api.h"

#define REGISTER_KERNEL_ALLREDUCE(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Allreduce").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AllreduceOp<type>)

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

template <typename T>
class AllreduceOp : public OpKernel {
public:
  explicit AllreduceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    RabitInit(0, nullptr);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
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
}

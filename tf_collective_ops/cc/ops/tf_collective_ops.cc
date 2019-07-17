#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Allreduce")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("Broadcast")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, float32, float64}")
    .Input("tensor: T")
    .Input("rank: int32")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

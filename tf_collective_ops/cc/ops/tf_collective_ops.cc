#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Allreduce")
    .Attr("T: list({int8, uint8, int32, uint32, int64, uint64, float32, float64})")
    .Input("n_tensors: uint32")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("Broadcast")
    .Attr("T: list({int8, uint8, int32, uint32, int64, uint64, float32, float64})")
    .Input("rank: int32")
    .Input("n_tensors: uint32")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });

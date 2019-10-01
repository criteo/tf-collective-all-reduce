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

REGISTER_OP("Allgather")
    .Attr("T: list({int8, uint8, int32, uint32, int64, uint64, float32, float64})")
    .Input("n_tensors: uint32")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (size_t i = 1; i < c->num_inputs(); i++) {
        shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(i), 0, c->UnknownDim(), &output));
        c->set_output(i-1, output);
      }
      return Status::OK();
    });

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow custom op example."""

from __future__ import absolute_import

from tf_collective_ops.python.ops.tf_collective_ops import allreduce, broadcast
from tf_collective_ops.python.ops.broadcast_variables_hook import BroadcastGlobalVariablesHook
from tf_collective_ops.python.ops.optimizer_wrapper import DistributedOptimizer 

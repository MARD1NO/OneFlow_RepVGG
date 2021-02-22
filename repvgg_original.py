"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Get original RepVGG output.
"""
from Classification.cnns.repvggmodel import RepVGG_A0
import oneflow as flow
import oneflow.typing as tp
import numpy as np


@flow.global_function()
def test_original_net(x: tp.Numpy.Placeholder(shape=(1, 3, 224, 224))) -> tp.Numpy:
    out = RepVGG_A0(x, None, deploy=False)
    return out


# Load the snapshot.
flow.load_variables(flow.checkpoint.get('./snapshot_epoch_1'))
x = np.ones(shape=(1, 3, 224, 224))
original_out = test_original_net(x)
np.save('original_net', original_out)

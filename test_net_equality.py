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

Evaluate the error.
"""
import numpy as np

original_output = np.load('./original_net.npy')
fused_output = np.load('fused_net.npy')


original_flatten = original_output.flatten()
fused_flatten = fused_output.flatten()
abs_val = 0
for i in range(len(original_flatten)):
    # Compute the accumulate abs value.
    abs_val += np.abs(original_flatten[i]-fused_flatten[i])
    # Compute the Maximum abs value.
    # abs_val = max(np.abs(original_flatten[i]-fused_flatten[i]), abs_val)

print("Abs Error is: ", abs_val)

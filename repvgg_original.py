import numpy as np
import os
import re

weight_folder = "snapshot_epoch_90"

weight_folder_list = os.listdir(weight_folder)

# for folder in weight_folder_list:
#     print(folder)

sample_weight_folder = "RepVGGstage_3_17"

sample_weight_list = []
for folder in weight_folder_list:
    # RepVGGstage_3_12_3x3_convbnlayer-moving_variance
    # -> ['RepVGGstage', '3', '12', '3x3', 'convbnlayer-moving', 'variance']
    folder_name_split = folder.split('_')
    if len(folder_name_split) <= 2:
        # Like RepVGGclassify_bias-momentum
        # print(folder_name_split)
        continue
    format_folder_name = folder_name_split[0] + "_" + folder_name_split[1] + "_" + folder_name_split[2]
    if format_folder_name == sample_weight_folder:
        sample_weight_list.append(folder)
    # raise Exception("Stop")

CONV_WEIGHT_NAME = "conv_conv_layer_weight"
CONV_KERNEL_SIZE_NAME = ["1x1", "3x3"]
CONV_BRANCH_BN_PARAM_NAME = ["conv_bn_layer-moving_mean",
                             "conv_bn_layer-moving_variance",
                             "conv_bn_layer-beta",
                             "conv_bn_layer-gamma"]
IDENTITY_BRANCH_BN_PARAM_NAME = ["identity_bn_layer-moving_mean",
                                 "identity_bn_layer-moving_variance",
                                 "identity_bn_layer-beta",
                                 "identity_bn_layer-gamma"]
print(sample_weight_list)

CONV_WEIGHT_3X3_LIST = []
CONV_WEIGHT_3X3_BN_LIST = []
CONV_WEIGHT_1X1_LIST = []
CONV_WEIGHT_1X1_BN_LIST = []

kernel_weight_name = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_WEIGHT_NAME
kernel_bn_mean = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[0]
kernel_bn_var = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[1]
kernel_bn_beta = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[2]
kernel_bn_gamma = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[3]

conv_weight_data_name = os.path.join(weight_folder, kernel_weight_name, 'out')
conv_bn_mean_data_name = os.path.join(weight_folder, kernel_bn_mean, 'out')
conv_bn_var_data_name = os.path.join(weight_folder, kernel_bn_var, 'out')
conv_bn_beta_data_name = os.path.join(weight_folder, kernel_bn_beta, 'out')
conv_bn_gamma_data_name = os.path.join(weight_folder, kernel_bn_gamma, 'out')

with open(conv_weight_data_name, 'rb') as f:
    conv_weight_data = f.read()
    conv_weight_data = np.frombuffer(conv_weight_data, dtype=np.float32)

with open(conv_bn_mean_data_name, 'rb') as f:
    conv_bn_mean_data = f.read()
    conv_bn_mean_data = np.frombuffer(conv_bn_mean_data, dtype=np.float32)

with open(conv_bn_var_data_name, 'rb') as f:
    conv_bn_var_data = f.read()
    conv_bn_var_data = np.frombuffer(conv_bn_var_data, dtype=np.float32)

with open(conv_bn_gamma_data_name, 'rb') as f:
    conv_bn_gamma_data = f.read()
    conv_bn_gamma_data = np.frombuffer(conv_bn_gamma_data, dtype=np.float32)

with open(conv_bn_beta_data_name, 'rb') as f:
    conv_bn_beta_data = f.read()
    conv_bn_beta_data = np.frombuffer(conv_bn_beta_data, dtype=np.float32)

conv_weight_data = np.reshape(conv_weight_data, newshape=(192, 192, 3, 3))
# conv_bn_mean_data = np.reshape(conv_bn_mean_data, newshape=(1, 192, 1, 1))
# conv_bn_var_data = np.reshape(conv_bn_var_data, newshape=(1, 192, 1, 1))
# conv_bn_gamma_data = np.reshape(conv_bn_gamma_data, newshape=(1, 192, 1, 1))

# Use bias add did not need to reshape
# conv_bn_beta_data = np.reshape(conv_bn_beta_data, newshape=(1, 192, 1, 1))

print(conv_weight_data.shape)  # (192,192,3,3)
print(conv_bn_mean_data.shape)  # 192

import oneflow as flow
import oneflow.typing as tp


# @flow.global_function()
# def test_original_conv_bn(x: tp.Numpy.Placeholder(shape=(1, 192, 64, 64))) -> tp.Numpy:
#     weight_shape = (192, 192, 3, 3)
#     _conv_weight = flow.get_variable(
#         name="weight",
#         shape=weight_shape,
#         initializer=flow.ones_initializer()
#     )
#     conv_out = flow.nn.conv2d(x, _conv_weight, 1, (0, 0, 1, 1), groups=1, name="conv2d")
#     bias_shape = (192,)
#     bn_gamma = flow.get_variable(
#         name="gamma",
#         shape=(1, 192, 1, 1),
#         initializer=flow.ones_initializer()
#     )
#     bn_bias = flow.get_variable(
#         name="bias",
#         shape=bias_shape,
#         initializer=flow.ones_initializer()
#     )
#     bn_mean = flow.get_variable(
#         name="mean",
#         shape=(1, 192, 1, 1),
#         initializer=flow.ones_initializer()
#     )
#     bn_var = flow.get_variable(
#         name="var",
#         shape=(1, 192, 1, 1),
#         initializer=flow.ones_initializer()
#     )
#     return flow.nn.bias_add(
#         bn_gamma*(conv_out-bn_mean)/flow.math.sqrt(bn_var), bn_bias
#     )


H = W = 16


@flow.global_function()
def test_original_conv_bn(x: tp.Numpy.Placeholder(shape=(1, 192, H, W))) -> tp.Numpy:
    weight_shape = (192, 192, 3, 3)
    _conv_weight = flow.get_variable(
        name="weight",
        shape=weight_shape,
        initializer=flow.ones_initializer()
    )
    conv_out = flow.nn.conv2d(x, _conv_weight, 1, (0, 0, 1, 1), groups=1, name="conv2d")

    bn_conv = flow.layers.batch_normalization(
        conv_out, axis=1, epsilon=1e-5, name="bn_layer", training=False, trainable=False,
    )
    return bn_conv

# check = flow.train.CheckPoint()
# check.init()
# flow.load_variables({"weight": conv_weight_data,
#                      "gamma": conv_bn_gamma_data,
#                      "bias": conv_bn_beta_data,
#                      "mean": conv_bn_mean_data,
#                      "var": conv_bn_var_data})
#
# # x = np.random.randn(1, 192, 64, 64).astype(np.float32)
# x = np.ones(shape=(1, 192, 64, 64)).astype(np.float32)
#
# fused_out = test_original_conv_bn(x)
# print(fused_out.shape)
# print(fused_out)
# np.save('original_conv_bn.npz', fused_out)


check = flow.train.CheckPoint()
check.init()
flow.load_variables({"weight": conv_weight_data,
                     "bn_layer-gamma": conv_bn_gamma_data,
                     "bn_layer-beta": conv_bn_beta_data,
                     "bn_layer-moving_mean": conv_bn_mean_data,
                     "bn_layer-moving_variance": conv_bn_var_data})
# x = np.random.randn(1, 192, 64, 64).astype(np.float32)
H = W = 16
x = np.ones(shape=(1, 192, H, W)).astype(np.float32)

fused_out = test_original_conv_bn(x)
print(fused_out.shape)
print(fused_out)
np.save('original_conv_bnv3', fused_out)
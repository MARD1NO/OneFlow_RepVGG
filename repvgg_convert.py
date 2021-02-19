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


def fuse_conv_bn(weight_folder, weight_folder_prefix, kernel_size):
    """
    Fuse BN into ConvLayer
    :param weight_folder: The name for the folder, example: snapshot_epoch_90
    :param weight_folder_prefix: The prefix name for the folder, example: RepVGGstage_3_17
    :param kernel_size: The kernel size, if it is 1x1, it will be pad to 3x3
    :return: Fused kernel weight, Fused kernel bias
    """
    if kernel_size == 1:
        KERNEL_SIZE_NAME = CONV_KERNEL_SIZE_NAME[0]  # "1x1"
    elif kernel_size == 3:
        KERNEL_SIZE_NAME = CONV_KERNEL_SIZE_NAME[1]  # "3x3"
    else:
        raise Exception("Wrong Conv Kernel size to Fuse!!!!")

    # Concat strings to get the folder name.
    kernel_weight_name = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_WEIGHT_NAME
    kernel_bn_mean = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[0]
    kernel_bn_var = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[1]
    kernel_bn_beta = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[2]
    kernel_bn_gamma = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[3]

    # concat "out" to get the weight file.
    conv_weight_data_name = os.path.join(weight_folder, kernel_weight_name, 'out')
    conv_bn_mean_data_name = os.path.join(weight_folder, kernel_bn_mean, 'out')
    conv_bn_var_data_name = os.path.join(weight_folder, kernel_bn_var, 'out')
    conv_bn_beta_data_name = os.path.join(weight_folder, kernel_bn_beta, 'out')
    conv_bn_gamma_data_name = os.path.join(weight_folder, kernel_bn_gamma, 'out')

    # Read data from weight file.
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

    # Reshape the conv weight.
    if kernel_size == 1:
        conv_weight_data = np.reshape(conv_weight_data, newshape=(192, 192, 1, 1))
        # Padding to 3x3 Kernel.
        conv_weight_data = np.pad(conv_weight_data, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
    elif kernel_size == 3:
        conv_weight_data = np.reshape(conv_weight_data, newshape=(192, 192, 3, 3))

    # Get the std
    conv_bn_std_data = np.sqrt(conv_bn_var_data)
    # Reshape BN Gamma as (out_channels, 1, 1, 1)
    conv_bn_t_data = np.reshape(conv_bn_gamma_data / conv_bn_std_data, newshape=(-1, 1, 1, 1))
    # Fuse BN to Conv weight
    conv_weight_fused_data = conv_weight_data * conv_bn_t_data
    conv_bn_fused_beta_data = conv_bn_beta_data - conv_bn_mean_data * conv_bn_gamma_data / conv_bn_std_data

    return conv_weight_fused_data, conv_bn_fused_beta_data


# kernel_3x3_weight_name = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_WEIGHT_NAME
# kernel_3x3_bn_mean = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[0]
# kernel_3x3_bn_var = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[1]
# kernel_3x3_bn_beta = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[2]
# kernel_3x3_bn_gamma = sample_weight_folder + "_" + CONV_KERNEL_SIZE_NAME[1] + "_" + CONV_BRANCH_BN_PARAM_NAME[3]
#
# conv_weight_data_name = os.path.join(weight_folder, kernel_3x3_weight_name, 'out')
# conv_bn_mean_data_name = os.path.join(weight_folder, kernel_3x3_bn_mean, 'out')
# conv_bn_var_data_name = os.path.join(weight_folder, kernel_3x3_bn_var, 'out')
# conv_bn_beta_data_name = os.path.join(weight_folder, kernel_3x3_bn_beta, 'out')
# conv_bn_gamma_data_name = os.path.join(weight_folder, kernel_3x3_bn_gamma, 'out')
#
# print(conv_weight_data_name)
#
# with open(conv_weight_data_name, 'rb') as f:
#     conv_weight_data = f.read()
#     conv_weight_data = np.frombuffer(conv_weight_data, dtype=np.float32)
#
# with open(conv_bn_mean_data_name, 'rb') as f:
#     conv_bn_mean_data = f.read()
#     conv_bn_mean_data = np.frombuffer(conv_bn_mean_data, dtype=np.float32)
#
# with open(conv_bn_var_data_name, 'rb') as f:
#     conv_bn_var_data = f.read()
#     conv_bn_var_data = np.frombuffer(conv_bn_var_data, dtype=np.float32)
#
# with open(conv_bn_gamma_data_name, 'rb') as f:
#     conv_bn_gamma_data = f.read()
#     conv_bn_gamma_data = np.frombuffer(conv_bn_gamma_data, dtype=np.float32)
#
# with open(conv_bn_beta_data_name, 'rb') as f:
#     conv_bn_beta_data = f.read()
#     conv_bn_beta_data = np.frombuffer(conv_bn_beta_data, dtype=np.float32)
#
# conv_weight_data = np.reshape(conv_weight_data, newshape=(192, 192, 3, 3))
# conv_bn_std_data = np.sqrt(conv_bn_var_data + 1e-5)
# conv_bn_t_data = np.reshape(conv_bn_gamma_data / conv_bn_std_data, newshape=(-1, 1, 1, 1))
# conv_weight_fused_data = conv_weight_data * conv_bn_t_data
# conv_bn_fused_beta_data = conv_bn_beta_data - conv_bn_mean_data * conv_bn_gamma_data / conv_bn_std_data


# print(conv_weight_data.shape)  # (192,192,3,3)
# print(conv_bn_mean_data.shape)  # 192
# print(conv_bn_std_data.shape)  # 192
# print(conv_bn_t_data.shape)  # (192,1,1,1)
# print(conv_bn_beta_data.shape)  # 192
#
# print(conv_weight_fused_data.shape)  # 192, 192, 3, 3
# print(conv_bn_fused_beta_data.shape)  # 192

conv_weight_fused_data, conv_bn_fused_beta_data = fuse_conv_bn(weight_folder, sample_weight_folder, 3)

import oneflow as flow
import oneflow.typing as tp


@flow.global_function()
def test_fused_conv(x: tp.Numpy.Placeholder(shape=(1, 192, 64, 64))) -> tp.Numpy:
    weight_shape = (192, 192, 3, 3)
    _conv_weight = flow.get_variable(
        name="weight",
        shape=weight_shape,
        initializer=flow.ones_initializer()
    )
    conv_out = flow.nn.conv2d(x, _conv_weight, 1, (0, 0, 1, 1), groups=1, name="conv2d")
    bias_shape = (192,)
    conv_bias = flow.get_variable(
        name="bias",
        shape=bias_shape,
        initializer=flow.ones_initializer()
    )
    conv_add_bias = flow.nn.bias_add(conv_out, bias=conv_bias)
    return conv_add_bias


# check = flow.train.CheckPoint()
# check.init()
flow.load_variables({"weight": conv_weight_fused_data,
                     "bias": conv_bn_fused_beta_data})

# x = np.random.randn(1, 192, 64, 64).astype(np.float32)
x = np.ones(shape=(1, 192, 64, 64)).astype(np.float32)

fused_out = test_fused_conv(x)
print(fused_out.shape)
print(fused_out)
np.save('fused_conv_bnv2.npz', fused_out)

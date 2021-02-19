import numpy as np
import os

weight_folder = "snapshot_epoch_90"

weight_folder_list = os.listdir(weight_folder)

# for folder in weight_folder_list:
#     print(folder)

sample_weight_folder = "RepVGGstage_3_17"
# sample_weight_folder = "RepVGGstage_1_2"


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
CONVWEIGHT_1X1_BN_LIST = []


def bn_fuse(weight_folder, weight_folder_prefix, in_channel, groups=1):
    """
    Fuse BN as a ConvLayer with constant weight.
    :param weight_folder: The name for the folder, example: snapshot_epoch_90
    :param weight_folder_prefix: The prefix name for the folder, example: RepVGGstage_3_17
    :param kernel_size: The kernel size, if it is 1x1, it will be pad to 3x3
    :return: Fused kernel weight, Fused kernel bias
    """
    input_dim = in_channel // groups
    # Because Identity ensure inchannel equal to outchannel.
    kernel_value = np.zeros(shape=(in_channel, input_dim, 3, 3), dtype=np.float32)

    # Identity equal to a unit matrix
    for i in range(in_channel):
        kernel_value[i, i % input_dim, 1, 1] = 1

    # Concat strings to get the folder name.
    kernel_bn_mean = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[0]
    kernel_bn_var = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[1]
    kernel_bn_beta = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[2]
    kernel_bn_gamma = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[3]

    # concat "out" to get the weight file.
    bn_mean_data_name = os.path.join(weight_folder, kernel_bn_mean, 'out')
    bn_var_data_name = os.path.join(weight_folder, kernel_bn_var, 'out')
    bn_beta_data_name = os.path.join(weight_folder, kernel_bn_beta, 'out')
    bn_gamma_data_name = os.path.join(weight_folder, kernel_bn_gamma, 'out')

    # Read data from weight file.
    with open(bn_mean_data_name, 'rb') as f:
        bn_mean_data = f.read()
        bn_mean_data = np.frombuffer(bn_mean_data, dtype=np.float32)

    with open(bn_var_data_name, 'rb') as f:
        bn_var_data = f.read()
        bn_var_data = np.frombuffer(bn_var_data, dtype=np.float32)

    with open(bn_gamma_data_name, 'rb') as f:
        bn_gamma_data = f.read()
        bn_gamma_data = np.frombuffer(bn_gamma_data, dtype=np.float32)

    with open(bn_beta_data_name, 'rb') as f:
        bn_beta_data = f.read()
        bn_beta_data = np.frombuffer(bn_beta_data, dtype=np.float32)

    # Get the std
    bn_std_data = np.sqrt(bn_var_data + 1e-5)
    # Reshape BN Gamma as (out_channels, 1, 1, 1)
    bn_t_data = np.reshape(bn_gamma_data / bn_std_data, newshape=(-1, 1, 1, 1))
    # Fuse BN to Conv weight
    bn_fused_weight_data = kernel_value * bn_t_data
    bn_fused_beta_data = bn_beta_data - bn_mean_data * bn_gamma_data / bn_std_data

    return bn_fused_weight_data, bn_fused_beta_data


out_channel = 192
in_channel = 192

bn_fused_weight_data, bn_fused_beta_data = bn_fuse(weight_folder,
                                                   sample_weight_folder,
                                                   in_channel,
                                                   groups=1)

import oneflow as flow
import oneflow.typing as tp

H = W = 16


@flow.global_function()
def test_fused_conv(x: tp.Numpy.Placeholder(shape=(1, 192, H, W))) -> tp.Numpy:
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
flow.load_variables({"weight": bn_fused_weight_data,
                     "bias": bn_fused_beta_data})

# x = np.random.randn(1, 192, 64, 64).astype(np.float32)
x = np.ones(shape=(1, 192, H, W)).astype(np.float32)

fused_out = test_fused_conv(x)
print(fused_out.shape)
print(fused_out)
np.save('fused_bn2conv', fused_out)

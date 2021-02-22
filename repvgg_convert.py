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

Convert BN->CONV and CONV+BN->CONV
"""
import numpy as np
import os

# These name is designed according to the repvgg.py
# Be careful if you want to change it, it may cause some weight didn't convert in right way.

CONV_WEIGHT_NAME = "conv_layer_weight"
CONV_BIAS_NAME = "conv_layer_bias"

CONV_KERNEL_SIZE_NAME = ["1x1", "3x3"]
CONV_BRANCH_BN_PARAM_NAME = ["bn_layer-moving_mean",
                             "bn_layer-moving_variance",
                             "bn_layer-beta",
                             "bn_layer-gamma"]
IDENTITY_BRANCH_BN_PARAM_NAME = ["identity_bn_layer-moving_mean",
                                 "identity_bn_layer-moving_variance",
                                 "identity_bn_layer-beta",
                                 "identity_bn_layer-gamma"]
CLASSIFY_NAME = "RepVGGclassify"
CLASSIFY_BRANCH_PARAM_NAME = ["dense_weight", "dense_bias"]


def conv_bn_fuse(weight_folder, weight_folder_prefix, kernel_size, eps, in_channel, out_channel, groups):
    """
    Fuse BN into ConvLayer
    :param weight_folder: The name for the folder, example: snapshot_epoch_90
    :param weight_folder_prefix: The prefix name for the folder, example: RepVGGstage_3_17
    :param kernel_size: The kernel size, if it is 1x1, it will be pad to 3x3
    :param eps: The epsilon used in BN Layer variance
    :param in_channel: The num of input channel.
    :param out_channel: The num of output channel.
    :param groups: The group num of ConvLayer.
    :return: Fused kernel weight, Fused kernel bias
    """
    assert in_channel // groups, "The input channel should be divided by groups"

    if_identity = False
    if kernel_size == 1:
        KERNEL_SIZE_NAME = CONV_KERNEL_SIZE_NAME[0]  # "1x1"
    elif kernel_size == 3:
        KERNEL_SIZE_NAME = CONV_KERNEL_SIZE_NAME[1]  # "3x3"
    elif kernel_size == 0:
        if_identity = True
    else:
        raise Exception("Wrong Conv Kernel size to Fuse!!!!")

    # Concat strings to get the folder name.
    if not if_identity:
        # Conv weight name
        kernel_weight_name = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_WEIGHT_NAME
        kernel_bn_mean = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[0]
        kernel_bn_var = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[1]
        kernel_bn_beta = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[2]
        kernel_bn_gamma = weight_folder_prefix + "_" + KERNEL_SIZE_NAME + "_" + CONV_BRANCH_BN_PARAM_NAME[3]
    else:
        # Identity weight name, and identity weight will be constructed in below.
        kernel_bn_mean = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[0]
        kernel_bn_var = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[1]
        kernel_bn_beta = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[2]
        kernel_bn_gamma = weight_folder_prefix + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[3]

    # concat "out" to get the weight file.
    if not if_identity:
        conv_weight_data_name = os.path.join(weight_folder, kernel_weight_name, 'out')

    conv_bn_mean_data_name = os.path.join(weight_folder, kernel_bn_mean, 'out')
    conv_bn_var_data_name = os.path.join(weight_folder, kernel_bn_var, 'out')
    conv_bn_beta_data_name = os.path.join(weight_folder, kernel_bn_beta, 'out')
    conv_bn_gamma_data_name = os.path.join(weight_folder, kernel_bn_gamma, 'out')

    # Read data from weight file.
    if not if_identity:
        # Conv branch has its weight
        with open(conv_weight_data_name, 'rb') as f:
            conv_weight_data = f.read()
            conv_weight_data = np.frombuffer(conv_weight_data, dtype=np.float32)
    else:
        # Identity branch:
        input_dim = in_channel // groups
        # Because Identity ensure inchannel equal to outchannel.
        conv_weight_data = np.zeros(shape=(in_channel, input_dim, 3, 3), dtype=np.float32)

        # Identity equal to a unit matrix
        for i in range(in_channel):
            conv_weight_data[i, i % input_dim, 1, 1] = 1

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
        # Padding to 3x3 Kernel.
        conv_weight_data = np.reshape(conv_weight_data, newshape=(out_channel, in_channel // groups, 1, 1))
        conv_weight_data = np.pad(conv_weight_data, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
    elif kernel_size == 3:
        conv_weight_data = np.reshape(conv_weight_data, newshape=(out_channel, in_channel // groups, 3, 3))

    # Get the std
    conv_bn_std_data = np.sqrt(conv_bn_var_data + eps)
    # Reshape BN Gamma as (out_channels, 1, 1, 1)
    conv_bn_t_data = np.reshape(conv_bn_gamma_data / conv_bn_std_data, newshape=(-1, 1, 1, 1))
    # Fuse BN to Conv weight
    conv_weight_fused_data = conv_weight_data * conv_bn_t_data
    conv_bn_fused_beta_data = conv_bn_beta_data - conv_bn_mean_data * conv_bn_gamma_data / conv_bn_std_data

    return conv_weight_fused_data, conv_bn_fused_beta_data


def block_convert(weight_folder, weight_folder_prefix, eps, in_channel, out_channel, groups, stride):
    """
    Convert RepVGG Block
    :param weight_folder: The name for the folder, example: snapshot_epoch_90
    :param weight_folder_prefix: The prefix name for the folder, example: RepVGGstage_3_17
    :param eps: The epsilon used in BN Layer variance
    :param in_channel: The num of input channel.
    :param out_channel: The num of output channel.
    :param groups: The num of Group ConvLayer.
    :param stride: The stride of Block. We will convert different branch according to the stride.
    :return:
    """
    conv_weight_1x1, conv_bias_1x1 = conv_bn_fuse(weight_folder,
                                                  weight_folder_prefix,
                                                  1,
                                                  eps,
                                                  in_channel,
                                                  out_channel,
                                                  groups=groups)
    conv_weight_3x3, conv_bias_3x3 = conv_bn_fuse(weight_folder,
                                                  weight_folder_prefix,
                                                  3,
                                                  eps,
                                                  in_channel,
                                                  out_channel,
                                                  groups=groups)
    if stride == 1:
        # Convert 1x1, 3x3 conv+bn and identity+bn.
        identity_weight, identity_bias = conv_bn_fuse(weight_folder,
                                                      weight_folder_prefix,
                                                      0,
                                                      eps,
                                                      in_channel,
                                                      out_channel,
                                                      groups=groups)
        fused_weight = conv_weight_1x1 + conv_weight_3x3 + identity_weight
        fused_bias = conv_bias_1x1 + conv_bias_3x3 + identity_bias
    elif stride == 2:
        # Convert 1x1, 3x3 conv+bn.
        fused_weight = conv_weight_1x1 + conv_weight_3x3
        fused_bias = conv_bias_1x1 + conv_bias_3x3
    else:
        raise Exception("Unsupported stride, only support 1 or 2")

    return fused_weight, fused_bias


def linear_convert(weight_folder, in_channel, out_channel):
    """
    Convert Linear Layer
    :param weight_folder: The name for the folder, example: snapshot_epoch_90
    :param in_channel: The input channel.
    :param out_channel: The output channel.
    :return: return weight and bias.
    """
    classify_weight_name = CLASSIFY_NAME + "_" + CLASSIFY_BRANCH_PARAM_NAME[0]
    classify_bias_name = CLASSIFY_NAME + "_" + CLASSIFY_BRANCH_PARAM_NAME[1]

    # concat "out" to get the weight file.
    classify_weight_name = os.path.join(weight_folder, classify_weight_name, 'out')
    classify_bias_name = os.path.join(weight_folder, classify_bias_name, 'out')

    # Read data from weight file.
    with open(classify_weight_name, 'rb') as f:
        classify_weight_data = f.read()
        classify_weight_data = np.frombuffer(classify_weight_data, dtype=np.float32)

    with open(classify_bias_name, 'rb') as f:
        classify_bias_data = f.read()
        classify_bias_data = np.frombuffer(classify_bias_data, dtype=np.float32)

    # Reshape the Dense weight.
    classify_weight_data = np.reshape(classify_weight_data, newshape=(out_channel, in_channel))

    return classify_weight_data, classify_bias_data


def weight_convert(weight_folder, num_blocks, num_filters, classify_nums, groupwise_layers, eps):
    # Convert the conv branch.
    if groupwise_layers is not None:
        groups_map = {l: 1 for l in groupwise_layers}
    else:
        groups_map = {}

    foldername_hashlist = {}
    for stage_idx in range(len(num_blocks) + 1):
        if stage_idx == 0:
            # Stage 0 only contains 1 layer
            layer_idx = 0
            layer_name = "RepVGGstage" + "_" + str(stage_idx) + "_" + str(layer_idx)
            stride = 2
            groups = groups_map.get(layer_idx, 1)
            in_channel = 3
            out_channel = int(num_filters[0])
            conv_fused_weight, conv_fused_bias = block_convert(weight_folder,
                                                               layer_name,
                                                               eps=eps,
                                                               in_channel=in_channel,
                                                               out_channel=out_channel,
                                                               groups=groups,
                                                               stride=stride)
            conv_weight_name = layer_name + "_3x3_" + CONV_WEIGHT_NAME
            conv_bias_name = layer_name + "_3x3_" + CONV_BIAS_NAME
            foldername_hashlist.update({conv_weight_name: conv_fused_weight})
            foldername_hashlist.update({conv_bias_name: conv_fused_bias})
        else:
            for layer_idx in range(num_blocks[stage_idx - 1]):
                if layer_idx == 0:
                    stride = 2
                    in_channel = int(num_filters[int(stage_idx) - 1])
                    out_channel = int(num_filters[int(stage_idx)])
                else:
                    stride = 1
                    in_channel = int(num_filters[int(stage_idx)])
                    out_channel = in_channel

                groups = groups_map.get(layer_idx, 1)
                # Add prevent idx and stage 0 layer idx is 1
                cur_layer_idx = layer_idx + sum(num_blocks[0: (stage_idx - 1)]) + 1
                # Special process.
                layer_name = "RepVGGstage" + "_" + str(stage_idx) + "_" + str(cur_layer_idx)

                conv_fused_weight, conv_fused_bias = block_convert(weight_folder,
                                                                   layer_name,
                                                                   eps=eps,
                                                                   in_channel=in_channel,
                                                                   out_channel=out_channel,
                                                                   groups=groups,
                                                                   stride=stride)
                conv_weight_name = layer_name + "_3x3_" + CONV_WEIGHT_NAME
                conv_bias_name = layer_name + "_3x3_" + CONV_BIAS_NAME
                foldername_hashlist.update({conv_weight_name: conv_fused_weight})
                foldername_hashlist.update({conv_bias_name: conv_fused_bias})

    # Convert Final Dense Layer.
    linear_weight_name = CLASSIFY_NAME + "_" + CLASSIFY_BRANCH_PARAM_NAME[0]
    linear_bias_name = CLASSIFY_NAME + "_" + CLASSIFY_BRANCH_PARAM_NAME[1]

    linear_weight, linear_bias = linear_convert(weight_folder, in_channel=int(num_filters[-1]),
                                                out_channel=classify_nums)
    foldername_hashlist.update({linear_weight_name: linear_weight})
    foldername_hashlist.update({linear_bias_name: linear_bias})

    return foldername_hashlist


# The RepVGG settings, you can also find in `repvgg.py`.
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

width_multiplier = [0.75, 0.75, 0.75, 2.5]
blocks_list = [2, 4, 14, 1]
filters_list = [min(64, int(64 * width_multiplier[0])), 64 * width_multiplier[0], 128 * width_multiplier[1],
                256 * width_multiplier[2], 512 * width_multiplier[3]]


# ====== The code in Below is test code. ======
# We choose `snapshot_epoch_1` as example.
weight_folder_name = "snapshot_epoch_1"

foldername_hashlist = weight_convert(weight_folder_name, blocks_list, filters_list, 1000, None, eps=1e-5)


from repvgg import RepVGG_A0
import oneflow as flow
import oneflow.typing as tp


@flow.global_function()
def test_fused_net(x: tp.Numpy.Placeholder(shape=(1, 3, 224, 224))) -> tp.Numpy:
    # Set deploy as True.
    out = RepVGG_A0(x, None, deploy=True)
    return out


# Load the converted weight.
flow.load_variables(foldername_hashlist)
x = np.ones(shape=(1, 3, 224, 224))
fused_out = test_fused_net(x)
# Save the output of Fused Net.
np.save('fused_net', fused_out)

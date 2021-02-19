import oneflow as flow
import oneflow.typing as tp
import numpy as np
import os


weight_folder = "snapshot_epoch_90"

sample_weight_folder = "RepVGGstage_3_17"

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

CONV_WEIGHT_3X3_LIST = []
CONV_WEIGHT_3X3_BN_LIST = []
CONV_WEIGHT_1X1_LIST = []
CONV_WEIGHT_1X1_BN_LIST = []

kernel_bn_mean = sample_weight_folder + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[0]
kernel_bn_var = sample_weight_folder + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[1]
kernel_bn_beta = sample_weight_folder + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[2]
kernel_bn_gamma = sample_weight_folder + "_" + IDENTITY_BRANCH_BN_PARAM_NAME[3]

bn_mean_data_name = os.path.join(weight_folder, kernel_bn_mean, 'out')
bn_var_data_name = os.path.join(weight_folder, kernel_bn_var, 'out')
bn_beta_data_name = os.path.join(weight_folder, kernel_bn_beta, 'out')
bn_gamma_data_name = os.path.join(weight_folder, kernel_bn_gamma, 'out')


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

H = W = 16


@flow.global_function()
def test_bn(x: tp.Numpy.Placeholder(shape=(1, 192, H, W))) -> tp.Numpy:
    bn = flow.layers.batch_normalization(
        x, axis=1, epsilon=1e-5, name="bn_layer", training=False, trainable=False,
    )
    return bn


flow.load_variables({"bn_layer-gamma": bn_gamma_data,
                     "bn_layer-beta": bn_beta_data,
                     "bn_layer-moving_mean": bn_mean_data,
                     "bn_layer-moving_variance": bn_var_data})
# x = np.random.randn(1, 192, 64, 64).astype(np.float32)
H = W = 16
x = np.ones(shape=(1, 192, H, W)).astype(np.float32)

fused_out = test_bn(x)
# print(fused_out.shape)
# print(fused_out)
np.save('original_bn', fused_out)

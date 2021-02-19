import numpy as np

original_output = np.load('./original_conv_bnv2.npz.npy')
# fused_output = np.load('./fused_conv_bn.npz.npy')
fused_output = np.load('./fused_conv_bnv2.npz.npy')

original_flatten = original_output.flatten()
fused_flatten = fused_output.flatten()

abs_val = 0
for i in range(331776):
    abs_val += np.abs(original_flatten[i]-fused_flatten[i])
    # abs_val = np.abs(original_flatten[i]-fused_flatten[i])
    #
    # if abs_val > 1e-5:
    #     print("Wrong ! ")
    #     print("Original is: ", original_flatten[i])
    #     print("Fused is: ", fused_flatten[i])
print(abs_val)
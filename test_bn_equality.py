import numpy as np

original_bn = np.load('./original_bn.npy')
fused_bn2conv = np.load('./fused_bn2conv.npy')

original_bn_flatten = original_bn.flatten()
fused_bn2conv_flatten = fused_bn2conv.flatten()
print(len(original_bn_flatten))
print(len(fused_bn2conv_flatten))

abs_val = 0


for i in range(len(original_bn)):
    abs_val += np.abs(original_bn_flatten[i]-fused_bn2conv_flatten[i])
    # abs_val += original_flatten[i]-fused_flatten[i]

    # abs_val = np.abs(original_flatten[i]-fused_flatten[i])
    #
    # if abs_val > 1e-5:
    #     print("Wrong ! ")
    #     print("Original is: ", original_flatten[i])
    #     print("Fused is: ", fused_flatten[i])
print(abs_val)
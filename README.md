# README

OneFlow 搭建RepVGG

# 转换脚本

参考 `repvgg_convert.py`

里面设置的名字是根据网络搭建的variable名设定的，请勿随意修改，会导致最终转换不对齐

# 原始脚本

参考`repvgg_original.py`

主要功能是加载训练好的原始模型，得到输出，以便和分支融合后的输出做对比

# 模型分支融合

首先模型经过训练，得到原始的权重。然后调用转换脚本 `repvgg_convert.py` 将权重进行分支，CONV+BN融合

### 精度列表

| RepVGGA0 | 72.69 |
| -------- | ----- |
|          |       |
|          |       |
|          |       |
|          |       |
|          |       |


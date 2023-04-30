# R-PolyNets on ImageNet

<!-- [ALGORITHM] -->

In this folder, you will find the code and pretrained model of R-PolyNets on ImageNet. The code and pretrained model of D-PolyNets on ImageNet will be available soon.


## Implemenation Details

Please follow [mmclassification](https://github.com/open-mmlab/mmclassification) to set up the training environment. Our models are trained by a single server with eight V100 GPUs.

All other training details follow the standard [configuration](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py).


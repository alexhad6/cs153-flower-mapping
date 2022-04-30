from deeplab.models import DeepLabV2_ResNet101_MSC
from deeplab.models.deeplabv2 import _ASPP
from deeplab.utils import DenseCRF

def ASPP(n_classes):
    ch = [64 * 2 ** p for p in range(6)]
    atrous_rates=[6, 12, 18, 24]
    return _ASPP(ch[5], n_classes, atrous_rates)

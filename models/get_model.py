from .vgg_rep import *
from .resnet_rep import *
from .wrn_rep import *
from .resnet import *

def get_model(model_name, num_classes):
    if model_name=='rep_vgg_3':
        return CVGG13_3(num_classes)
    elif model_name=='rep_vgg_4':
        return CVGG13_4(num_classes)
    elif model_name=='rep_vgg_5':
        return CVGG13_5(num_classes)
    elif model_name=='rep_vgg_6':
        return CVGG13_6(num_classes)
    elif model_name=='rep_vgg_7':
        return CVGG13_7(num_classes)
    elif model_name=='rep_vgg_8':
        return CVGG13_8(num_classes)
    elif model_name=='rep_vgg_9':
        return CVGG13_9(num_classes)
    elif model_name=='resnet20_38':
        return resnet20_38(num_classes)
    elif model_name=='resnet32_62':
        return resnet32_62(num_classes)
    elif model_name=='resnet44_86':
        return resnet44_86(num_classes)
    elif model_name=='resnet56_110':
        return resnet56_110(num_classes)
    elif model_name=='resnet110_218':
        return resnet110_218(num_classes)
    elif model_name=='resnet20':
        return resnet20(num_classes)
    elif model_name=='resnet32':
        return resnet32(num_classes)
    elif model_name=='resnet44':
        return resnet44(num_classes)
    elif model_name=='resnet56':
        return resnet56(num_classes)
    elif model_name=='resnet110':
        return resnet110(num_classes)
    elif model_name=='wrn_16_1':
        return wrn_16_1(num_classes)
    elif model_name=='wrn_16_4':
        return wrn_16_4(num_classes)
    elif model_name=='wrn_16_8':
        return wrn_16_8(num_classes)
    elif model_name=='wrn_16_10':
        return wrn_16_10(num_classes)
from .vgg_rep import *
from .resnet_rep import *
# from .group_equivariant import *

def get_model(model_name, num_classes, args=None):
    if model_name=='rep_vgg_4':
        return CVGG11_4(num_classes)
    elif model_name=='rep_vgg_5':
        return CVGG11_5(num_classes)
    elif model_name=='rep_vgg_6':
        return CVGG11_6(num_classes)
    elif model_name=='rep_vgg_7':
        return CVGG11_7(num_classes)
    elif model_name=='rep_vgg_8':
        return CVGG11_8(num_classes)
    elif model_name=='rep_vgg_9':
        return CVGG11_9(num_classes)
    elif model_name=='rep_vgg_10':
        return CVGG11_10(num_classes)
    elif model_name[:3]=='VGG':
        return VGG(model_name, num_classes)
    elif model_name=='resnet_16_1':
        return resnet_rep(num_classes, 1, args)
    elif model_name=='resnet_16_4':
        return resnet_rep(num_classes, 4, args)
    elif model_name=='resnet_16_8':
        return resnet_rep(num_classes, 8, args)
    elif model_name=='resnet_16_10':
        return resnet_rep(num_classes, 10, args)
    elif model_name=='c16':
        return c16(num_classes)
    elif model_name=='c8':
        return c8(num_classes)
    elif model_name=='c4':
        return c4(num_classes)
    elif model_name=='baseline':
        return baseline(num_classes)
    elif model_name=='c16_rep':
        return c16_rep(num_classes)
    elif model_name=='c8_rep':
        return c8_rep(num_classes)
    elif model_name=='c4_rep':
        return c4_rep(num_classes)
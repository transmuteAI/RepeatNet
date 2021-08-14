from .vgg_rep import *
from .resnet_rep import rresnetcifar, rresnet
from .resnet import resnetcifar, resnet
from .group_equivariant import *

def get_model(model_name, num_classes, args=None):
    if model_name=='rep_vgg_4':
        return CVGG11_4(num_classes, args)
    elif model_name=='rep_vgg_5':
        return CVGG11_5(num_classes, args)
    elif model_name=='rep_vgg_6':
        return CVGG11_6(num_classes, args)
    elif model_name=='rep_vgg_7':
        return CVGG11_7(num_classes, args)
    elif model_name=='rep_vgg_8':
        return CVGG11_8(num_classes, args)
    elif model_name=='rep_vgg_9':
        return CVGG11_9(num_classes, args)
    elif model_name=='rep_vgg_10':
        return CVGG11_10(num_classes, args)
    elif model_name[:3]=='VGG':
        return VGG(model_name, num_classes)
    elif model_name=='rresnet_cifar_16_1':
        return rresnetcifar(num_classes, 1, args)
    elif model_name=='rresnet_cifar_16_2':
        return rresnetcifar(num_classes, 2, args)
    elif model_name=='rresnet_cifar_16_4':
        return rresnetcifar(num_classes, 4, args)
    elif model_name=='rresnet_cifar_16_8':
        return rresnetcifar(num_classes, 8, args)
    elif model_name[:15]=='resnet_cifar_16':
        return resnetcifar(model_name, num_classes, args)
    elif model_name=='rresnet_18_1':
        return rresnet(num_classes, 1, args)
    elif model_name=='rresnet_18_2':
        return rresnet(num_classes, 2, args)
    elif model_name=='rresnet_18_4':
        return rresnet(num_classes, 4, args)
    elif model_name=='rresnet_18_8':
        return rresnet(num_classes, 8, args)
    elif model_name[:9]=='resnet_18':
        return resnet(model_name, num_classes, args)
    elif model_name=='c16':
        return c16(num_classes)
    elif model_name=='c8':
        return c8(num_classes)
    elif model_name=='c4':
        return c4(num_classes)
    elif model_name=='baseline':
        return baseline(num_classes)
    elif model_name=='c16_rep':
        return c16_rep(num_classes, args)
    elif model_name=='c8_rep':
        return c8_rep(num_classes, args)
    elif model_name=='c4_rep':
        return c4_rep(num_classes, args)

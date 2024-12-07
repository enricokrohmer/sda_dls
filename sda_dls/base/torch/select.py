# LICENSE
# This file was extracted from
#  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `sda_dls/base/LICENSE` for copyright attribution and LICENSE


from torch import nn

def extract_name_kwargs(obj):
    if obj is None:
        return None, {}
    elif isinstance(obj, str):
        return obj, {}
    
    name = obj.name
    if obj.get('args'):
        kwargs = obj.args
    else:
        kwargs = {}
    return name, kwargs


def get_norm_layer(norm_config, features):
    name, kwargs = extract_name_kwargs(norm_config)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)


def get_norm_layer_1D(norm_config, features):
    name, _ = extract_name_kwargs(norm_config)
    
    if name == 'batch':
        return nn.BatchNorm1d(features)
    elif name == 'instance':
        return nn.InstanceNorm1d(features)
    elif name == 'layer':
        return nn.LayerNorm((features,))
    elif name == None:
        return nn.Identity()
    else:
        raise NotImplementedError(f'norm [{norm_config}] is not implemented for domain classifier')


def get_norm_layer_fn(norm_config):
    return lambda features : get_norm_layer(norm_config, features)


def get_activ_layer(activ_config):
    name, kwargs = extract_name_kwargs(activ_config)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(inplace = True, **kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(inplace = True, **kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)

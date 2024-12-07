from typing import Tuple
import os
import torch

def module_weights_from_pl_ckpt(
    model: torch.nn.Module, 
    ckpt: Tuple[str, str],
) -> None:
    path, name = ckpt
    if not os.path.exists(path):
        raise ValueError("Checkpoint does not exist: %s" % path)
        
    state_dict = torch.load(path, map_location='cpu')['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if k.startswith(name)}
    state_dict = {k.replace(name + '.', ''): v for k, v in state_dict.items()}
    return model.load_state_dict(state_dict, strict=False)


@torch.no_grad()
def update_average_model(average_model, model, momentum):
    params = list(model.parameters())
    avg_params = list(average_model.parameters())
    
    if len(params) != len(avg_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
                
    for param, avg_param in zip(params, avg_params):
        tmp = avg_param - param
        tmp.mul_(1.0 - momentum)
        avg_param.sub_(tmp)


def domain_wise_metrics(metric_dict, nc_source):
        metric_dict_A = {k + '_source': 
            torch.mean(v[0:nc_source]) for k, v in metric_dict.items()}
        metric_dict_B = {k + '_target': 
            torch.mean(v[nc_source:]) for k, v in metric_dict.items()}
        metric_dict = metric_dict_A | metric_dict_B
        return metric_dict

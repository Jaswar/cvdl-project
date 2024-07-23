import os
import torch
import yaml
from models.sceneRepresentation import Scene
from dataset.dataset import DynamicPixelDataset, get_split_dynamic_pixel_data, ImageDataset_CVDL
from util.initialValues import estimate_initial_values, estimate_initial_vals_pendulum
import matplotlib.pyplot as plt
from torchvision import utils
from util.util import compute_psnr, compute_iou
import numpy as np


def main():
    experiments_path = os.path.join(
        os.path.abspath(''),
        'PhysParamInference',
        'experiments',
        '2024-07-23'
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Using device: {device}')

    for experiment in os.listdir(experiments_path):
        experiment_path = os.path.join(experiments_path, experiment)
        parameter_estimates = {}
        for instance in os.listdir(experiment_path):
            instance_path = os.path.join(experiment_path, instance)

            config_path = os.path.join(instance_path, '.hydra','config.yaml')
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            model = Scene(**cfg['scene']['background'])
            if experiment == 'pendulum_cvdl':
                model.add_pendulum(**cfg['ode'], **cfg['scene']['local_representation'])
            elif experiment == 'sliding_block_cvdl':
                model.add_slidingBlock(alpha=torch.tensor(0.0), 
                       p0=torch.tensor([0.0, 0.0]),
                       **cfg['ode'], 
                       **cfg['scene']['local_representation'])
            elif experiment == 'bouncing_ball_drop_cvdl':
                model.add_BouncingBallDrop_CVDL(elasticity=torch.tensor(0.0), 
                       p0=torch.tensor([0.0, 0.0]),
                       **cfg['ode'], 
                       **cfg['scene']['local_representation'])
            elif experiment == 'ball_throw_cvdl':
                model.add_thrownObject(p0=torch.tensor([0.0, 0.0]),
                       **cfg['ode'], 
                       **cfg['scene']['local_representation'])
            else:
                raise ValueError(f'Unsupported experiment {experiment}')

            path_ckpt = os.path.join(instance_path, 'ckpt.pth')
            model.load_state_dict(torch.load(path_ckpt))

            for name, param in model.local_representation.ode.named_parameters():
                if name not in parameter_estimates:
                    parameter_estimates[name] = []
                parameter_estimates[name].append(param.item())
        
        print(f'Parameter estimates for {experiment}')
        for param, estimates in parameter_estimates.items():
            mean = round(np.mean(estimates), 2)
            std = round(np.std(estimates), 2)
            print(f'{param}={mean} (std. {std})')

if __name__ == '__main__':
    main()
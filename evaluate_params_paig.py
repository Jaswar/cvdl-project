import os
import numpy as np


def main():
    base_path = os.path.join(
        os.path.abspath(''),
        'paig',
        'runners',
    )

    experiments_params = {}
    for experiment_instance in os.listdir(base_path):
        if not experiment_instance[-1].isnumeric():
            continue
        
        experiment = experiment_instance[:-2]
        experiment_instance_path = os.path.join(base_path, experiment_instance)
        for file in os.listdir(experiment_instance_path):
            file_path = os.path.join(experiment_instance_path, file)
            if file.endswith('.txt') and file != 'log.txt':
                with open(file_path, 'r') as f:
                    lines = f.read().split()
                param_value = float(lines[-1])
                param_name = file.replace('.txt', '')

                if experiment not in experiments_params:
                    experiments_params[experiment] = {}
                if param_name not in experiments_params[experiment]:
                    experiments_params[experiment][param_name] = []
                experiments_params[experiment][param_name].append(param_value)
    
    for experiment, params in experiments_params.items():
        print(f'Parameter estimates for {experiment}:')
        for param, values in params.items():
            mean = round(np.mean(values), 2)
            std = round(np.std(values), 2)
            print(f'{param}={mean} (std. {std})')
        print()



if __name__ == '__main__':
    main()

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
from src.train import train_model
from src.eval import evaluate_model

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'])
    # args = parser.parse_args()

    train = True
    eval = True

    models = {
        'params':           {'center': 0.4, 'diag': 0.3, 'quat': 0.3, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0},
        'corners':          {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 1.0, 'iou': 0.0},
        'canonical':        {'center': 0.3, 'diag': 0.1, 'quat': 0.0, 'canonical': 0.6, 'corner': 0.0, 'iou': 0.0},
        'iou':              {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.0, 'iou': 1.0},
        # 'params_corners':   {'center': 0.2, 'diag': 0.1, 'quat': 0.2, 'canonical': 0.0, 'corner': 0.5, 'iou': 0.0},
        # 'params_iou':       {'center': 0.2, 'diag': 0.1, 'quat': 0.2, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.5},
        # 'corners_iou':      {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.5, 'iou': 0.5}
    }
    
    if train:

        for model_name, loss_weights in models.items():
            print(f"Training {model_name} with loss weights: {loss_weights}")
            train_model(model_name=model_name, loss_weights=loss_weights)

    elif eval:

        for model_name in models.keys():
            print(f"Evaluating {model_name} with loss weights: {models[model_name]}")
            evaluate_model(model_name=f'{model_name}_best', loss_weights=models[model_name])
import torch
import argparse
from finetune import ModifiedVGG16Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    model = torch.load(args.model_path).cuda()
    print model.features
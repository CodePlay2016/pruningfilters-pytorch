import torch
import argparse
from finetune import ModifiedVGG16Model
import dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./log/prune-2018-06-05_175834_f/model_pruned")
    args = parser.parse_args()
    return args

def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for i, (batch, label) in enumerate(data_loader):
        batch = batch.cuda()
        output = model(Variable(batch, volatile=True))
        pred = output.data.max(1)[1]
        correct += pred.cpu().eq(label).sum()
        total += label.size(0)
    acc = float(correct) / total
    print("Test Accuracy :%.4f" % (acc))
    return acc


if __name__ == '__main__':
    args = get_args()
    model = torch.load(args.model_path).cuda()
    print model
    _, _, data_loader = dataset.train_valid_test_loader("train")
    acc = test(model, data_loader)
    print(acc)

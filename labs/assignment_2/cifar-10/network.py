import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        pass
        # ----------TODO------------
        # define a network 
        # ----------TODO------------

    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        # ----------TODO------------

        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 

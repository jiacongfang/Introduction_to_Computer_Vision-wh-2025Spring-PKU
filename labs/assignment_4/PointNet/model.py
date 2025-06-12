from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    """
    The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
    Args:
    d: the dimension of the global feature, default is 1024.
    segmentation: whether to perform segmentation, default is True.
    """

    def __init__(self, segmentation=False, d=1024):
        super(PointNetfeat, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the feature extractor. ##
        ## 3 -> 64 -> 128 -> d
        ## ------------------------------------------- ##
        self.segmentation = segmentation
        self.d = d
        self.layer_1 = nn.Conv1d(3, 64, kernel_size=1)
        self.layer_2 = nn.Conv1d(64, 128, kernel_size=1)
        self.layer_3 = nn.Conv1d(128, d, kernel_size=1)

        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.batch_norm_3 = nn.BatchNorm1d(d)

    def forward(self, x):
        """
        If segmentation == True
            return the concatenated global feature and local feature. # (B, d+64, N)
        If segmentation == False
            return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
        Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        """
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x = x.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
        B, D, N = x.size()

        x = F.relu(self.batch_norm_1(self.layer_1(x)))  # (B, 64, N)

        local_feature = x  # store the first layer output for segmentation

        x = F.relu(self.batch_norm_2(self.layer_2(x)))  # (B, 128, N)
        x = self.batch_norm_3(self.layer_3(x))  # (B, d, N)

        # max_pooling
        gloabl_feature = torch.max(x, dim=-1, keepdim=False)[0]  # (B, d)

        if self.segmentation:
            x = torch.cat(
                [local_feature, gloabl_feature.unsqueeze(-1).repeat(1, 1, N)], dim=1
            )  # (B, d+64, N)
            return x
        else:
            return gloabl_feature, x.permute(0, 2, 1)  # (B, d), (B, N, d)


class PointNetCls1024D(nn.Module):
    """
    The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
    Args:
    k: the number of classes, default is 2.
    """

    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## 1024 -> 512 -> 256 -> k
        ## ------------------------------------------- ##
        self.pointnet_feat = PointNetfeat(segmentation=False, d=1024)
        self.k = k
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.k),
        )
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        """
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x, per_point_feature = self.pointnet_feat(x)
        x = self.layers(x)
        x = self.softmax(x)  # (B, k)
        return x, per_point_feature  # (B, k), (B, N, d=1024)


class PointNetCls256D(nn.Module):
    """
    The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
    Args:
    k: the number of classes, default is 2.
    """

    def __init__(self, k=2):
        super(PointNetCls256D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## 256 -> 128 -> k
        ## ------------------------------------------- ##
        self.pointnet_feat = PointNetfeat(segmentation=False, d=256)
        self.k = k
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.k),
        )
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        """
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x, per_point_feature = self.pointnet_feat(x)
        x = self.layers(x)
        x = self.softmax(x)  # (B, k)
        return x, per_point_feature  # (B, k), (B, N, d=256)


class PointNetSeg(nn.Module):
    """
    The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
    Args:
    k: the number of classes, default is 2.
    """

    def __init__(self, k=2):
        super(PointNetSeg, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the segmentation head. ##
        ## 64+d -> 512 -> 256 -> 128 -> k
        ## ------------------------------------------- ##
        self.k = k
        self.pointnet_feat = PointNetfeat(segmentation=True, d=1024)
        self.layers = nn.Sequential(
            nn.Conv1d(64 + 1024, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.k, kernel_size=1),
        )
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Input:
            x: the input point cloud. # (B, N, 3)
        Output:
            the log softmax of the segmentation result. # (B, N, k)
        """
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x = self.pointnet_feat(x)  # (B, 64+d, N)
        x = self.layers(x)  # (B, k, N)
        x = x.permute(0, 2, 1)  # (B, N, k)

        # apply log softmax
        x = self.softmax(x)  # (B, N, k)
        return x

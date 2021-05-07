# ========================================================================================
# Geometric transformation (affine & tps)
# Author: Jingwei Qu
# Date: 2020/05/18 12:46
# ========================================================================================

import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import cv2


def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))


class ComposedGeometricTnf(object):
    """
    Composed geometric transfromation (affine+tps)
    """

    def __init__(self, tps_grid_size=3, tps_reg_factor=0, tps_dim=36, out_h=240, out_w=240, offset_factor=1.0,
                 padding_crop_factor=None, use_cuda=True):
        self.padding_crop_factor = padding_crop_factor
        self.affTnf = GeometricTnf(out_h=out_h, out_w=out_w, geometric_model='affine',
                                   offset_factor=offset_factor if padding_crop_factor is None else padding_crop_factor,
                                   use_cuda=use_cuda)

        self.tpsTnf = GeometricTnf(out_h=out_h, out_w=out_w, geometric_model='tps', tps_grid_size=tps_grid_size,
                                   tps_reg_factor=tps_reg_factor, tps_dim=tps_dim,
                                   offset_factor=offset_factor if padding_crop_factor is None else 1.0,
                                   use_cuda=use_cuda)

    def __call__(self, image_batch, theta_aff, theta_aff_tps, use_cuda=True):
        sampling_grid_aff = self.affTnf(image_batch=None, theta_batch=theta_aff.view(-1, 2, 3),
                                        return_sampling_grid=True, return_warped_image=False)

        sampling_grid_aff_tps = self.tpsTnf(image_batch=None, theta_batch=theta_aff_tps,
                                            return_sampling_grid=True, return_warped_image=False)

        if self.padding_crop_factor is not None:
            sampling_grid_aff_tps = sampling_grid_aff_tps * self.padding_crop_factor

        # put 1e10 value in region out of bounds of sampling_grid_aff
        in_bound_mask_aff = ((sampling_grid_aff[:, :, :, 0] > -1) * (sampling_grid_aff[:, :, :, 0] < 1) * (sampling_grid_aff[:, :, :, 1] > -1) * (sampling_grid_aff[:, :, :, 1] < 1)).unsqueeze(3)
        in_bound_mask_aff = in_bound_mask_aff.expand_as(sampling_grid_aff)
        sampling_grid_aff = torch.mul(in_bound_mask_aff.float(), sampling_grid_aff)
        sampling_grid_aff = torch.add((in_bound_mask_aff.float() - 1) * (1e10), sampling_grid_aff)

        # compose transformations
        sampling_grid_aff_tps_comp = F.grid_sample(sampling_grid_aff.permute(0, 3, 1, 2), sampling_grid_aff_tps).permute(0, 2, 3, 1)

        # put 1e10 value in region out of bounds of sampling_grid_aff_tps_comp
        in_bound_mask_aff_tps = ((sampling_grid_aff_tps[:, :, :, 0] > -1) * (sampling_grid_aff_tps[:, :, :, 0] < 1) * (sampling_grid_aff_tps[:, :, :, 1] > -1) * (sampling_grid_aff_tps[:, :, :, 1] < 1)).unsqueeze(3)
        in_bound_mask_aff_tps = in_bound_mask_aff_tps.expand_as(sampling_grid_aff_tps_comp)
        sampling_grid_aff_tps_comp = torch.mul(in_bound_mask_aff_tps.float(), sampling_grid_aff_tps_comp)
        sampling_grid_aff_tps_comp = torch.add((in_bound_mask_aff_tps.float() - 1) * (1e10), sampling_grid_aff_tps_comp)

        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid_aff_tps_comp)

        return warped_image_batch

class GeometricTnf(object):
    """

    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )

    """
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, use_cuda=True, tps_grid_size=3, tps_reg_factor=0,
                 tps_dim=36, offset_factor=None):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        self.geometric_model = geometric_model
        self.offset_factor = offset_factor
        self.tps_dim = tps_dim

        if geometric_model == 'affine' and offset_factor is None:
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'affine' and offset_factor is not None:
            self.gridGen = AffineGridGenV2(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'tps' and tps_dim == 18:
            self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, grid_size=tps_grid_size, reg_factor=tps_reg_factor,
                                      use_cuda=use_cuda)
        elif geometric_model == 'tps' and tps_dim == 36:
            self.gridGen = TpsGridGen2(out_h=out_h, out_w=out_w, grid_size=tps_grid_size, reg_factor=tps_reg_factor,
                                       use_cuda=use_cuda)
        elif geometric_model == 'tps' and tps_dim == 24:
            self.gridGen = TpsGridGen3(out_h=out_h, out_w=out_w, grid_size=tps_grid_size, reg_factor=tps_reg_factor,
                                       use_cuda=use_cuda)
        elif geometric_model == 'tps' and tps_dim == 32:
            self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, grid_size=4, reg_factor=tps_reg_factor, use_cuda=use_cuda)

        if offset_factor is not None:
            self.gridGen.grid_X = self.gridGen.grid_X / offset_factor
            self.gridGen.grid_Y = self.gridGen.grid_Y / offset_factor

        # theta_identity.shape: (1, 2, 3), mainly use as affine transformation parameters for
        # (1) resize the image from original size to (480, 640) when initializing the dataset
        # (2) crop the image from (480. 640) to (240, 240) when generating the training image pairs
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0, out_h=None, out_w=None,
                 return_warped_image=True, return_sampling_grid=False):
        # padding_factor and crop_factor are used for grid
        # image_batch.shape: (batch_size, 3, H, W)
        if image_batch is None:
            b = 1
        else:
            b = image_batch.shape[0]    # Get batch_size
        # Use theta_identity as affine transformation parameters for
        # (1) resize the image from original size to (480, 640) when initializing the dataset
        # (2) crop the image from (480. 640) to (240, 240) when generating the training image pairs
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b, 2, 3).contiguous()
            theta_batch.requires_grad = False
            # theta_batch = Variable(theta_batch, requires_grad=False)

        # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h != self.out_h or out_w != self.out_w):
            if self.geometric_model == 'affine' and self.offset_factor is None:
                self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'affine' and self.offset_factor is not None:
                self.gridGen = AffineGridGenV2(out_h=out_h, out_w=out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'tps' and self.tps_dim == 18:
                self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'tps' and self.tps_dim == 36:
                self.gridGen = TpsGridGen2(out_h=out_h, out_w=out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'tps' and self.tps_dim == 24:
                self.gridGen = TpsGridGen3(out_h=out_h, out_w=out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'tps' and self.tps_dim == 32:
                self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, grid_size=4, use_cuda=self.use_cuda)

        else:
            gridGen = self.gridGen

        # Generate the grid for geometric transformation (affine or tps) with the given theta (theta_batch)
        # theta is the parameters for geometric transformation from output image to input image
        # sampling_grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240)
        # 2 includes coordinates (x, y) in the input image (image_batch)
        # For (x, y) in sampling_grid[i][j] (ignore batch dim):
        # use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
        sampling_grid = gridGen(theta_batch)

        # Rescale (x, y) in grid (i.e. coordinates in the input image) with crop_factor and padding_factor
        if padding_factor != 1 or crop_factor != 1:
        # if abs(padding_factor - 1.0) > 1e-6 or abs(crop_factor - 1.0) > 1e-6:
            sampling_grid = sampling_grid * (padding_factor * crop_factor)
        if self.offset_factor is not None:
            sampling_grid = sampling_grid * self.offset_factor

        if return_sampling_grid and not return_warped_image:
            return sampling_grid

        # sample transformed image, warped_image_batch.shape: (batch_size, 3, out_h, out_h)
        # For (x, y) in sampling_grid[i][j] (ignore batch dim):
        # use pixel value in (x, y) of the image (image_batch) as the pixel value in (i, j) of the image (warped_image_batch)
        # (x, y) is float, use default bilinear interpolation to obtain the pixel value in (x, y)
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)

        if return_sampling_grid and return_warped_image:
            return warped_image_batch, sampling_grid

        return warped_image_batch

# Generate the grid for affine transformation with the given theta
# theta is the parameters for affine transformation from output image to input image
# grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240), 2 includes coordinates (x, y) in the input image
# For (x, y) in grid[i][j] (ignore batch dim):
# use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        # Reshape affine theta as (batch_size, 2, 3), 6 parameters
        b = theta.size()[0]
        if not theta.size() == (b, 2, 3):
            theta = theta.view(-1, 2, 3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)

class AffineGridGenV2(Module):
    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(AffineGridGenV2, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is out_h * out_w
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # self.grid_X, self.grid_Y: size [1, out_w, out_h, 1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self, theta):
        # Reshape affine theta as (batch_size, 6), 6 parameters
        b = theta.size(0)
        if not theta.size() == (b, 6):
            theta = theta.view(b, 6)
            theta = theta.contiguous()

        t0 = theta[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1 = theta[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2 = theta[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3 = theta[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4 = theta[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5 = theta[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        X = expand_dim(self.grid_X, 0, b)
        Y = expand_dim(self.grid_Y, 0, b)
        Xp = X * t0 + Y * t1 + t2
        Yp = X * t3 + Y * t4 + t5

        return torch.cat((Xp, Yp), 3)

# Generate the grid for tps transformation with the given theta
# theta is the parameters for tps transformation from output image to input image
# grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240), 2 includes coordinates (x, y) in the input image
# For (x, y) in grid[i][j] (ignore batch dim):
# use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
class TpsGridGen(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor    # Regularized factor for matrix K
        self.use_cuda = use_cuda

        # Create grid in numpy, i.e. self.grid_X and self.grid_Y
        # self.grid.shape: (out_h, out_w, 3)
        # self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y, out_h)
        # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is out_h * out_w
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        self.grid_X = torch.Tensor(self.grid_X.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.Tensor(self.grid_Y.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # Initialize regular grid for control points P_i (self.P_X and self.P_Y), 3 * 3
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is 3 * 3
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            # P_X.shape and P_Y.shape: (9, 1)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.Tensor(P_X.astype(np.float32))
            P_Y = torch.Tensor(P_Y.astype(np.float32))
            # self.Li.shape: (1, 12, 12)
            # self.Li = Variable(self.compute_L_inverse(P_X, P_Y).unsqueeze(0), requires_grad=False)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.Li.requires_grad = False
            # self.P_X.shape and self.P_Y.shape: (1, 1, 1, 1, 9)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_X.requires_grad = False
            self.P_Y.requires_grad = False
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):

        # Generate the warped grid for tps transformation with the given theta and the grid
        # theta.shape: (batch_size, 18) for tps
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        # warped_grid.shape: (batch_size, out_h, out_w, 2)
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        # X.shape and Y.shape: (9, 1)
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K, Xmat.shape and Ymax.shape: (9. 9)
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        # Distance squared matrix, P_dist_squared.shape: (9, 9)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        # P_dist_squared = P_dist_squared + 1e-6  # make diagonal 1 to avoid NaN in log computation
        # K.shape: (9, 9), P.shape: (9, 3), L.shape: (12, 12)
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # Add regularized term for matrix K
        if self.reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * self.reg_factor
        # construct matrix L
        O = torch.Tensor(N, 1).fill_(1)
        Z = torch.Tensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        # Li is inverse matrix of L, Li.shape: (12, 12)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    # Generate the warped grid for tps transformation with the given theta and the grid
    def apply_transformation(self, theta, points):
        # theta.shape: (batch_size, 18) for tps
        # points.shape: (batch_size, out_h, out_w, 2), for loss: (batch_size, 1, 400, 2)
        # theta.shape becomes (batch_size, 18, 1, 1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        # Q_X.shape and Q_Y.shape: (batch_size, 9, 1)
        # Vertical axis is X, horizontal axis is Y
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        # P_X.shape and P_Y.shape: (1, out_h, out_w, 1, 9)
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # TPS consists of an affine part and a non-linear part
        # compute weigths for non-linear part
        # W_X.shape and W_Y.shape: (batch_size, 9, 1)
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N], i.e. (batch_size, out_h, out_w, 1, 9)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        # A_X.shape and A_Y.shape: (batch_size, 3, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3], i.e. (batch_size, out_h, out_w, 1, 3)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points_X_for_summation.shape and points_Y_for_summation.shape: (batch_size, H, W, 1, 9)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N], i.e.(1, out_h, out_w, 1, 9)
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        # points_X_batch.shape and points_Y_batch.shape: (batch_size, out_h, out_w, 1)
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        # points_X_prime.shape and points_Y_prime.shape: (batch_size, out_h, out_w, 1)
        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        # Return grid.shape: (batch_size, out_h, out_w, 2)
        return torch.cat((points_X_prime, points_Y_prime), 3)

# Generate the grid for tps transformation with the given theta
# theta is the parameters for tps transformation from output image to input image
# grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240), 2 includes coordinates (x, y) in the input image
# For (x, y) in grid[i][j] (ignore batch dim):
# use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
# For tps-36dim
class TpsGridGen2(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen2, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor    # Regularized factor for matrix K
        self.use_cuda = use_cuda

        # Create grid in numpy, i.e. self.grid_X and self.grid_Y
        # self.grid.shape: (out_h, out_w, 3)
        # self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y, out_h)
        # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is out_h * out_w
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        self.grid_X = torch.Tensor(self.grid_X.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.Tensor(self.grid_Y.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        self.N = grid_size * grid_size

    def forward(self, theta):
        # Generate the warped grid for tps transformation with the given theta and the grid
        # theta.shape: (batch_size, 18) for tps
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        # warped_grid.shape: (batch_size, out_h, out_w, 2)
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        # X.shape and Y.shape: (b, 9, 1)
        batch_size = X.size()[0]
        N = X.size()[1]  # num of points (along dim 1)
        # construct matrix K, Xmat.shape and Ymax.shape: (b, 9, 9)
        Xmat = X.expand(batch_size, N, N)
        Ymat = Y.expand(batch_size, N, N)
        # Distance squared matrix, P_dist_squared.shape: (b, 9, 9)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(1, 2), 2) + torch.pow(Ymat - Ymat.transpose(1, 2), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        # P_dist_squared = P_dist_squared + 1e-6  # make diagonal 1 to avoid NaN in log computation
        # K.shape: (b, 9, 9), P.shape: (b, 9, 3), L.shape: (b, 12, 12)
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # Add regularized term for matrix K
        if self.reg_factor != 0:
            K += torch.eye(K.size(1), K.size(2)).unsqueeze(0).expand(batch_size, K.size(1), K.size(2)) * self.reg_factor
        # construct matrix L, L.shape: (b, 12, 12)
        O = torch.Tensor(batch_size, N, 1).fill_(1)
        Z = torch.Tensor(batch_size, 3, 3).fill_(0)
        P = torch.cat((O, X, Y), 2)
        L = torch.cat((torch.cat((K, P), 2), torch.cat((P.transpose(1, 2), Z), 2)), 1)
        # Li is inverse matrix of L, Li.shape: (b, 12, 12)
        # print(L[0])
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    # Generate the warped grid for tps transformation with the given theta and the grid
    def apply_transformation(self, theta, points):
        # theta.shape: (batch_size, 36) for tps
        # theta.shape becomes (batch_size, 36, 1, 1)
        # points.shape: (batch_size, points_h, points_w, 2), for loss: (batch_size, 1, 400, 2)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        # Q_X.shape, Q_Y.shape, P_X.shape, P_Y.shape: (batch_size, 9, 1)
        # Vertical axis is X, horizontal axis is Y
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:2*self.N, :, :].squeeze(3)
        P_X = theta[:, 2*self.N:3*self.N, :, :].squeeze(3)
        P_Y = theta[:, 3*self.N:, :, :].squeeze(3)

        Li = self.compute_L_inverse(P_X.cpu(), P_Y.cpu())
        # Li.requires_grad = False

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        # P_X.shape and P_Y.shape: (b, points_h, points_w, 1, 9)
        P_X = P_X.unsqueeze(3).unsqueeze(4).transpose(1, 4)
        P_Y = P_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4)
        P_X = P_X.expand((batch_size, points_h, points_w, 1, self.N))
        P_Y = P_Y.expand((batch_size, points_h, points_w, 1, self.N))

        # TPS consists of an affine part and a non-linear part
        # compute weigths for non-linear part
        # W_X.shape and W_Y.shape: (batch_size, 9, 1)
        W_X = torch.bmm(Li[:, :self.N, :self.N], Q_X)
        W_Y = torch.bmm(Li[:, :self.N, :self.N], Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N], i.e. (batch_size, out_h, out_w, 1, 9)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        # A_X.shape and A_Y.shape: (batch_size, 3, 1)
        A_X = torch.bmm(Li[:, self.N:, :self.N], Q_X)
        A_Y = torch.bmm(Li[:, self.N:, :self.N], Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3], i.e. (batch_size, out_h, out_w, 1, 3)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points_X_for_summation.shape and points_Y_for_summation.shape: (batch_size, H, W, 1, 9)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        delta_X = points_X_for_summation - P_X
        delta_Y = points_Y_for_summation - P_Y

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N], i.e.(1, out_h, out_w, 1, 9)
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        # points_X_batch.shape and points_Y_batch.shape: (batch_size, out_h, out_w, 1)
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        # points_X_prime.shape and points_Y_prime.shape: (batch_size, out_h, out_w, 1)
        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        # Return grid.shape: (batch_size, out_h, out_w, 2)
        return torch.cat((points_X_prime, points_Y_prime), 3)

    def apply_transformation_R(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)

        batch_size = theta.size()[0]
        P_X = theta[:, :self.N, :, :].squeeze(3)
        P_Y = theta[:, self.N:2*self.N, :, :].squeeze(3)
        Q_X = theta[:, 2*self.N:3*self.N, :, :].squeeze(3)
        Q_Y = theta[:, 3*self.N:, :, :].squeeze(3)

        Li = self.compute_L_inverse(P_X.cpu(), P_Y.cpu())

        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        P_X = P_X.unsqueeze(3).unsqueeze(4).transpose(1, 4)
        P_Y = P_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4)
        P_X = P_X.expand((batch_size, points_h, points_w, 1, self.N))
        P_Y = P_Y.expand((batch_size, points_h, points_w, 1, self.N))

        W_X = torch.bmm(Li[:, :self.N, :self.N], Q_X)
        W_Y = torch.bmm(Li[:, :self.N, :self.N], Q_Y)

        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        A_X = torch.bmm(Li[:, self.N:, :self.N], Q_X)
        A_Y = torch.bmm(Li[:, self.N:, :self.N], Q_Y)

        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        delta_X = points_X_for_summation - P_X
        delta_Y = points_Y_for_summation - P_Y

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)

# Generate the grid for tps transformation with the given theta
# theta is the parameters for tps transformation from output image to input image
# grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240), 2 includes coordinates (x, y) in the input image
# For (x, y) in grid[i][j] (ignore batch dim):
# use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
# For tps-24dim
class TpsGridGen3(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen3, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor    # Regularized factor for matrix K
        self.use_cuda = use_cuda

        # Create grid in numpy, i.e. self.grid_X and self.grid_Y
        # self.grid.shape: (out_h, out_w, 3)
        # self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y, out_h)
        # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is out_h * out_w
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        self.grid_X = torch.Tensor(self.grid_X.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.Tensor(self.grid_Y.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # Initialize regular grid for control points P_i (self.P_X and self.P_Y), 3 * 3
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is 3 * 3
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            # P_X.shape and P_Y.shape: (9, 1)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.Tensor(P_X.astype(np.float32))
            P_Y = torch.Tensor(P_Y.astype(np.float32))
            # self.P_X.shape and self.P_Y.shape: (1, 1, 1, 1, 9)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_X.requires_grad = False
            self.P_Y.requires_grad = False
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):

        # Generate the warped grid for tps transformation with the given theta and the grid
        # theta.shape: (batch_size, 18) for tps
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        # warped_grid.shape: (batch_size, out_h, out_w, 2)
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    # Generate the warped grid for tps transformation with the given theta and the grid
    def apply_transformation(self, theta, points):
        # theta.shape: (batch_size, 24) for tps, [0:6] for a ([0:3] x, [3:6] y), [6:] for w ([6:15] x [15:] y)
        # points.shape: (batch_size, out_h, out_w, 2), for loss: (batch_size, 1, 400, 2)
        # theta.shape becomes (batch_size, 18, 1, 1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # TPS consists of an affine part and a non-linear part
        # Weights for affine part: A_X.shape and A_Y.shape: (batch_size, 3, 1)
        # Weights for non-linear part: W_X.shape and W_Y.shape: (batch_size, 9, 1)
        A_X = theta[:, :3, :, :].squeeze(3)
        A_Y = theta[:, 3:6, :, :].squeeze(3)
        W_X = theta[:, 6:6+self.N, :, :].squeeze(3)
        W_Y = theta[:, 6+self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        # P_X.shape and P_Y.shape: (1, out_h, out_w, 1, 9)
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # reshape
        # W_X,W,Y: size [B,H,W,1,N], i.e. (batch_size, out_h, out_w, 1, 9)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3], i.e. (batch_size, out_h, out_w, 1, 3)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points_X_for_summation.shape and points_Y_for_summation.shape: (batch_size, H, W, 1, 9)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N], i.e.(1, out_h, out_w, 1, 9)
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        # points_X_batch.shape and points_Y_batch.shape: (batch_size, out_h, out_w, 1)
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        # points_X_prime.shape and points_Y_prime.shape: (batch_size, out_h, out_w, 1)
        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        # Return grid.shape: (batch_size, out_h, out_w, 2)
        return torch.cat((points_X_prime, points_Y_prime), 3)

# For tps-32dim
class TpsGridGen4(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=4, reg_factor=0, use_cuda=True):
        super(TpsGridGen4, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor    # Regularized factor for matrix K
        self.use_cuda = use_cuda

        # Create grid in numpy, i.e. self.grid_X and self.grid_Y
        # self.grid.shape: (out_h, out_w, 3)
        # self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y, out_h)
        # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is out_h * out_w
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        self.grid_X = torch.Tensor(self.grid_X.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.Tensor(self.grid_Y.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # Initialize regular grid for control points P_i (self.P_X and self.P_Y), 3 * 3
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is 3 * 3
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            # P_X.shape and P_Y.shape: (9, 1)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.Tensor(P_X.astype(np.float32))
            P_Y = torch.Tensor(P_Y.astype(np.float32))
            # self.Li.shape: (1, 12, 12)
            # self.Li = Variable(self.compute_L_inverse(P_X, P_Y).unsqueeze(0), requires_grad=False)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.Li.requires_grad = False
            # self.P_X.shape and self.P_Y.shape: (1, 1, 1, 1, 9)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_X.requires_grad = False
            self.P_Y.requires_grad = False
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):

        # Generate the warped grid for tps transformation with the given theta and the grid
        # theta.shape: (batch_size, 18) for tps
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        # warped_grid.shape: (batch_size, out_h, out_w, 2)
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        # X.shape and Y.shape: (9, 1)
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K, Xmat.shape and Ymax.shape: (9. 9)
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        # Distance squared matrix, P_dist_squared.shape: (9, 9)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        # P_dist_squared = P_dist_squared + 1e-6  # make diagonal 1 to avoid NaN in log computation
        # K.shape: (9, 9), P.shape: (9, 3), L.shape: (12, 12)
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # Add regularized term for matrix K
        if self.reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * self.reg_factor
        # construct matrix L
        O = torch.Tensor(N, 1).fill_(1)
        Z = torch.Tensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        # Li is inverse matrix of L, Li.shape: (12, 12)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    # Generate the warped grid for tps transformation with the given theta and the grid
    def apply_transformation(self, theta, points):
        # theta.shape: (batch_size, 18) for tps
        # points.shape: (batch_size, out_h, out_w, 2), for loss: (batch_size, 1, 400, 2)
        # theta.shape becomes (batch_size, 18, 1, 1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        # Q_X.shape and Q_Y.shape: (batch_size, 9, 1)
        # Vertical axis is X, horizontal axis is Y
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        # P_X.shape and P_Y.shape: (1, out_h, out_w, 1, 9)
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # TPS consists of an affine part and a non-linear part
        # compute weigths for non-linear part
        # W_X.shape and W_Y.shape: (batch_size, 9, 1)
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N], i.e. (batch_size, out_h, out_w, 1, 9)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        # A_X.shape and A_Y.shape: (batch_size, 3, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3], i.e. (batch_size, out_h, out_w, 1, 3)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points_X_for_summation.shape and points_Y_for_summation.shape: (batch_size, H, W, 1, 9)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N], i.e.(1, out_h, out_w, 1, 9)
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        # points_X_batch.shape and points_Y_batch.shape: (batch_size, out_h, out_w, 1)
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        # points_X_prime.shape and points_Y_prime.shape: (batch_size, out_h, out_w, 1)
        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        # Return grid.shape: (batch_size, out_h, out_w, 2)
        return torch.cat((points_X_prime, points_Y_prime), 3)
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from torch import nn
import os
import json
from plyfile import PlyData, PlyElement

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, optimizer_type="default"):
        self.magnitude=1
        self.position=torch.empty(0)
        self.orientation = torch.empty(0)
        self.scale=torch.empty(0)
        self.cov=torch.empty(0)
        self.albedo=torch.empty(0)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self.position], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "position"},
            {'params': [self.magnitude], 'lr': training_args.magnitude_lr, "name": "magnitude"},
            {'params': [self.scale], 'lr': training_args.scale_lr, "name": "scale"},
            {'params': [self.orientation], 'lr': training_args.orientation_lr, "name": "orientation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "position":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def load_gs(self,path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        gs_data=np.load(path)
        self.position = torch.tensor(gs_data["positions"], dtype=torch.float32, device=device)
        self.scale = torch.tensor(gs_data["scales"], dtype=torch.float32, device=device)
        self.scale*=25
        self.orientation = torch.tensor(gs_data["orientations"], dtype=torch.float32, device=device)
        self.orientation =torch.transpose(self.orientation,1,2)
        scale_mid = torch.zeros((self.scale.size(0), 3,3), dtype=torch.float32, device=device)
        scale_mid[:,0,0]=self.scale[:,0]
        scale_mid[:,1,1]=self.scale[:,1]
        scale_mid[:,2,2]=self.scale[:,2]
        #orientation_mid = torch.zeros((self.scale.size(0), 3,3), dtype=torch.float32, device=device)
        mid=torch.matmul(self.orientation,scale_mid)
        t_mid=torch.transpose(mid,1,2)
        self.cov=torch.matmul(mid,t_mid)
    
    def quaternion_to_rotation_matrix(self,quaternions):
        """
        Convert a sequence of quaternions to rotation matrices.
        
        Parameters:
            quaternions (np.ndarray): Input array of shape [n, 4], where each row is [w, x, y, z].
        
        Returns:
            np.ndarray: Output array of shape [n, 9], where each row is a flattened 3x3 rotation matrix.
        """
        # Normalize the quaternions to avoid numerical instability
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # Compute the rotation matrices
        rotation_matrices = np.stack([
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
        ], axis=-1).T.reshape(-1, 9)
        
        return rotation_matrices

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        '''
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        '''
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        orien=self.quaternion_to_rotation_matrix(rots)

        features_dc=features_dc.squeeze(2)
        print("xyz")
        print(xyz.shape)
        print("features_dc")
        print(features_dc.shape)
        print("opacities")
        print(opacities.shape)
        print("scales")
        print(scales.shape)
        print("orien")
        print(orien.shape)


        self.position = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.albedo = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True))
        self.magnitude = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self.scale = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.orientation = nn.Parameter(torch.tensor(orien, dtype=torch.float, device="cuda").requires_grad_(True))
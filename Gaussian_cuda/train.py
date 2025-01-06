import os
from PIL import Image
import numpy as np

import torch
import cuda_renderer
print(torch.__version__)
print(torch.__file__)     # 查看 PyTorch 安装路径
from scene.gaussian_model import GaussianModel
from scene import Scene
from diff_render import Diff_Render
from gaussian_renderer import render

def training(iterations,gt_image,env_path,out_path):
    gaussians = GaussianModel()

    load_path="/data/qinhaoran/processed/gs_new_bunny.npz"
    load_path1="/data/qinhaoran/point_cloud.ply"
    gaussians.load_ply(load_path1)
    
    #gaussians.load_gs(load_path)
    scene = Scene(gaussians,env_path)
    #gaussians.training_setup()

    first_iter =0

    for iteration in range(first_iter, iterations + 1):
        #gaussians.update_learning_rate(iteration)
        image = render(scene,gaussians,scene.env_rgbs)
    print(image.shape)
    image_cpu = image.cpu()  # 移动到 CPU
    image_np = image_cpu.numpy()  # 转换为 NumPy 数组
    image_np = (image_np * 255).astype(np.uint8)
    pic = Image.fromarray(image_np)
    pic.save(out_path)

        #Ll1 = l1_loss(image, gt_image)
        #loss = Ll1

if __name__ == "__main__":
    env_path="/home/qinhaoran/upload/IndoorEnvironmentHDRI001_8K-TONEMAPPED.jpg"
    out_path="/home/qinhaoran/cuda_output/output_image.png"
    #x=torch.empty(0)
    #print(cuda_renderer.render_c(x))
    gt_image=1
    training(0,gt_image,env_path,out_path)

'''
export TORCH_USE_RTLD_GLOBAL=1
alias python="python3"
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
''' 


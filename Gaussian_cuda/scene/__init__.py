import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from scene.gaussian_model import GaussianModel
from torchvision import transforms
from PIL import Image
import torch



class Scene:

    gaussians : GaussianModel

    def __init__(self,args : ModelParams,gaussians : GaussianModel,env_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.camera=Camera()
        self.env_path=env_path
        env_image = Image.open(env_path)
        transform = transforms.ToTensor()
        self.env_rgbs = transform(env_image).to(device) 
        self.env_rgbs = self.env_rgbs.permute(1, 2, 0)
        print("env")
        print(self.env_rgbs.shape)


        
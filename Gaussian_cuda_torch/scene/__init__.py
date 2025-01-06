from scene.gaussian_model import GaussianModel
from torchvision import transforms
from PIL import Image
import torch

class Camera:
    def init(self):
        self.resolution=1024
        self.pos=[1.5,1,1]


class Scene:

    gaussians : GaussianModel

    def __init__(self,gaussians : GaussianModel,env_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.camera=Camera()
        self.env_path=env_path
        env_image = Image.open(env_path)
        transform = transforms.ToTensor()
        self.env_rgbs = transform(env_image).to(device) 
        self.env_rgbs = self.env_rgbs.permute(1, 2, 0)
        print("env")
        print(self.env_rgbs.shape)


        
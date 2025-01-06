import torch
import math
from scene.gaussian_model import GaussianModel
from diff_render import Diff_Render

def render(camera,gsm:GaussianModel,env_rgbs):
    return Diff_Render.apply(gsm.position,gsm.scale,gsm.orientation,gsm.cov,gsm.magnitude,gsm.albedo,env_rgbs)
    
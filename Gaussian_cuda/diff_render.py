import torch
import cuda_renderer

class Diff_Render(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position,scale,orientation,cov,magnitude,albedo,env_rgbs):
        '''
        print("gsm.position type:", type(position))
        print("gsm.scale type:", type(scale))
        print("gsm.orientation type:", type(orientation))
        print("gsm.cov type:", type(cov))
        print("gsm.magnitude type:", type(magnitude))
        print("env_rgbs type:", type(env_rgbs))
        '''
        
        print("position")
        print(position)
        print("scale")
        print(scale)
        print("orientation")
        print(orientation)
        print("cov")
        print(cov.shape)
        
        output=cuda_renderer.render_c(position,scale,orientation,cov,magnitude,albedo,env_rgbs)
        #output = 1
        return output
    '''
    @staticmethod
    def backward(ctx, g_position,g_scale,g_orientation,g_magnitude):
        # 从上下文中加载前向计算时保存的变量
        input, = ctx.saved_tensors
        # 计算梯度：根据前向定义的 y = x^2, dy/dx = 2 * x
        grad_input = grad_output * 2 * input
        return grad_input
    '''
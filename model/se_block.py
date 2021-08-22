from torch import nn 

class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, ratio_reduce):
        super(SqueezeExcitation, self).__init__()
        reduced_dim = in_channels//ratio_reduce
        self.SEnet = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),  # C x 1 x 1 -> C/r x 1 x 1
            SiLU(),  # in original using ReLU
            nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),  # C/r x 1 x 1 -> C x 1 x 1
            nn.Sigmoid())
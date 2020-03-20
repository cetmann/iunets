import sys
import torch
import iunets
from iunets import networks
from iunets.networks import iUNet
import numpy as np

device="cuda"
n_channels = 24

# Create some meaningsless data
x = (torch.arange(n_channels*256*256*256).float().view(1,n_channels,256,256,256)/(n_channels*256*256*256))
x = x.contiguous().to(device)

# MemCNN deletes inputs, so if we want to use x later (in this case to
# check the inversion quality), we need to copy it. In this case, this reduces the number
x_orig=x.clone() 

model = iUNet(n_channels, # input channels or input shape, must be at least as large as slice_fraction
              dim=3, # 3D data input
              architecture=[10,10,10,10,10,10,10], # 7*10*2=140 convolutional layers
              create_module_fn=iunets.layers.create_standard_module,
              slice_fraction = 4, # Fraction of 
              learnable_downsampling=True, # Otherwise, 3D Haar wavelets are used
              disable_custom_gradient=False
              ).to(device)

y = model(x)
loss = torch.sum(y)

# Calculate the gradients of the sum of outputs with respect to the weights
loss.backward()
model.zero_grad()

# Check the inversion
z = model.inverse(y) 
diff=(x_orig-z)
print(torch.mean(diff**2))
import torchvision
import torch
import sys
sys.path.insert(0,'/home/ce377/my_research/iunet/pytorch/iunet_lib')
import iunets
from iunets import networks
from iunets.networks import iUNet

device="cuda"

# Load MNIST dataset
data = torchvision.datasets.MNIST(
    root='/data/septal/ce377/MNIST',
    train=True, transform=
        torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

data_loader = torch.utils.data.DataLoader(
    data,
    batch_size=1024,
    shuffle=True,
    num_workers=4)


# Number of channels that the data is increased to by a linear transform (needs to be at least 2)
n_blowup_channels = 16

# MNIST is 28-by-28. Pad to increase to 32-by-32, so that we can use invertible downsampling more than 2 times.
size = 32
padding = torch.nn.ReplicationPad2d(2).to(device)

# The dimensionality of the data
dim = 2

"""
Define the architecture: 
  increase the number of channels to n_blowup_channels
  invertible U-Net
  global average pooling
  dense layer
  softmax + NLL loss
"""
blowup_layer = torch.nn.Conv2d(1, n_blowup_channels, 1)
blowup_layer.weight.data = (torch.ones_like(blowup_layer.weight) / n_blowup_channels)
iunet = iUNet(n_blowup_channels,
              dim=dim, # 1D, 2D or 3D invertible U-Net?
              slice_fraction=2, # Should be dim**2 if we want to double the number of channels at every downsampling
              architecture=[2,2,2,2,2], 
              disable_custom_gradient=False, 
              create_module_fn=networks.create_standard_module, 
              learnable_downsampling=True # If false, use Haar wavelet transform instead
              )
pooling_layer = torch.nn.Conv2d(n_blowup_channels, n_blowup_channels, size, bias=False)#, groups=n_blowup_channels)
flatten = torch.nn.Flatten()
FC_layer = torch.nn.Linear(n_blowup_channels, 10)
model = torch.nn.Sequential(
    blowup_layer,
    iunet,
    pooling_layer,
    flatten,
    FC_layer).to(device)
model.zero_grad()
loss_curve = []
loss_fn = torch.nn.CrossEntropyLoss()



# Define the optimizer
model_weights = list(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=.000001)

# Train for 20 epochs
for epoch in range(20):
    for id, (x, y) in enumerate(data_loader):
        x, y = padding(x.to(device)), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        loss_curve.append(loss.cpu().detach().numpy())
        optimizer.step()
print("Done.")
# Imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

import iunets
from iunets.networks import iUNet
from iunets.baseline_networks import StandardUNet


import h5py
import os
import configparser
import argparse
import pathlib
import random



device='cuda'
size = 256

"""
PARSING SECTION -- load instructions from the INI-file.
"""

parser = argparse.ArgumentParser(
    description='Train an iUNet for learned post-processing.')
parser.add_argument('ini_file', type=str)
args = parser.parse_args()

ini_file = args.ini_file


# Load the desired specifications from the provided
# INI-file.
config = configparser.ConfigParser()
config.read(ini_file)
paths = config['PATHS']
hyperparameters = config['HYPERPARAMETERS']

# Currently unused LOGGING section
#logging = config['LOGGING']

# HYPERPARAMETERS
use_invertible_unet = hyperparameters.getboolean(
    'use_invertible_unet',
    True)

if not use_invertible_unet:
    print("Using non-invertible (standard) U-Net!")

# If the standard U-Net is used, do we want a skip connection,
# i.e. y=x+f(x), where f is the standard U-Net?
non_invertible_unet_with_skip_connection = hyperparameters.getboolean(
    'non_invertible_unet_with_skip_connection',
    True)


disable_custom_gradient = hyperparameters.getboolean(
    'disable_custom_gradient',
    True)

# Whether 
learnable = hyperparameters.getboolean(
    'learnable_downsampling',
    True)

initial_learning_rate = float(hyperparameters.get(
    'initial_learning_rate'))
print("initial_learning_rate",initial_learning_rate)
learning_rate_schedule = eval(hyperparameters.get(
    'learning_rate_schedule'))
learning_rate_epochs = [np.int(item[0]) for item in learning_rate_schedule]
learning_rate_multiplier = [np.float32(item[1]) for item in learning_rate_schedule] 
check_sorting = learning_rate_epochs[:]
check_sorting.sort()
assert(learning_rate_epochs == check_sorting)

batch_size = int(hyperparameters.get(
    'batch_size'))


architecture = [int(i)
                   for i in eval(
                       hyperparameters.get(
                           'architecture',
                           '[2,2,2,2]'
                       )
                   )
                  ]
slice_fraction = int(hyperparameters.get(
    'slice_fraction'))
n_blowup_channels = int(hyperparameters.get(
    'n_blowup_channels'))

weight_decay_parameter = float(hyperparameters.get(
    'weight_decay_parameter'))
num_epochs = int(hyperparameters.get(
    'num_epochs'))



# PATHS
base_name = os.path.splitext(
    os.path.basename(ini_file))[0]
if not use_invertible_unet:
    base_name += "_STANDARD_UNET"

tensorboard_logdir = paths.get(
    'tensorboard_logdir') + \
    base_name + '/'
folder = paths.get(
    'data_folder')
saved_model_folder = paths.get(
    'saved_model_folder')
saved_model_path = saved_model_folder + \
    base_name + '/model.pt' 


"""
END OF PARSING SECTION
"""


"""
START OF DATA SECTION
"""

# Data augmentation function
#! For segmentation experiments: Change this whole section. 
#! Adapt the axes for the reflections and rotations. Maybe add a DataLoader for batching.
def augment(image, 
            ground_truth):
    # Random rotations
    dim = len(image.shape)-2
    
    n_rotations = np.random.randint(3)
    if n_rotations:
        image = torch.rot90(image, n_rotations, [3,4])
        ground_truth = torch.rot90(ground_truth, n_rotations, [3,4])

    # Random flipping
    # List of all possible flip axes
    possible_flip_axes = [2,3,4]
    flip_axes = random.sample(possible_flip_axes, k=random.randint(0,3))
    if len(flip_axes)>0:
        torch.flip(image, flip_axes)
        torch.flip(ground_truth, flip_axes)
    
    return image, ground_truth

# Dataset object
class FoamDataset(Dataset):
    def __init__(self, file_list, device=None, augment=False):
        self.file_list = file_list
        self.augment = augment
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        f = h5py.File(self.file_list[idx], 'r')
        x = torch.tensor(f['reconstruction'][:])
        y = torch.tensor(f['volume'][:])
        f.close()
        
        # Add batch, channel dimension
        x = x.view(1, 1, *x.shape)
        y = y.view(1, 1, *y.shape)
        
        if self.augment:
            x, y = augment(x, y)
            
        if device is not None:
            x = x.to(device)
            y = y.to(device)
            
        return x.contiguous(), y.contiguous()

# Make sure folder ends with /
if folder[-1] != '/':
    folder += '/'

# Search for all files containing '_data.h5' in the specified folder and
# use 80% for training and 10% for validation.
# Testing of the remaining 10% not implemented yet, but is the same as validation.
all_files = sorted([folder + f for f in os.listdir(folder) if '_data.h5' in f])
train_filenames = all_files[int(.8*len(all_files)):]
validation_filenames = all_files[int(.8*len(all_files)):int(.9*len(all_files))]


train_data = FoamDataset(train_filenames,
                        device='cuda',
                        augment=True)

validation_data = FoamDataset(train_filenames,
                        device='cuda',
                        augment=False)

"""
END OF DATA SECTION
"""



"""
START OF ARCHITECTURE DEFINITION
"""
if use_invertible_unet:
    
    blowup_layer = torch.nn.Conv3d(1, n_blowup_channels, 3, padding=1)
    iunet = iUNet(n_blowup_channels, # input channels or input shape, must be at least as large as slice_fraction
              dim=3, # 3D data input
              architecture=architecture,
              create_module_fn=iunets.layers.create_standard_module,
              slice_fraction = 4,
              learnable_downsampling=True, # Otherwise, 3D Haar wavelets are used
              disable_custom_gradient=disable_custom_gradient
              )
    collapse_layer = nn.Conv3d(n_blowup_channels, 1, 1, padding=1) 
    #! For segmentation experiments:
    #! collapse_layer = nn.Conv3d(n_blowup_channels, n_classes, 1)
    
    
    
    # Initialize the blowup and collapse layers such that the whole iUNet is
    # initialized as the identity function.
    
    blowup_kernel = np.zeros((n_blowup_channels,1,3,3,3),dtype='float32')
    blowup_kernel[:,:,1,1,1] = 1. / n_blowup_channels
    blowup_layer.weight.data = torch.tensor(blowup_kernel).data
    
    collapse_kernel = np.zeros((1,n_blowup_channels,3,3,3),dtype='float32')
    collapse_kernel[:,:,1,1,1] = 1. / n_blowup_channels
    collapse_layer.weight.data = torch.tensor(collapse_kernel).data
    
    model = nn.Sequential(blowup_layer,
                         iunet,
                         collapse_layer).to(device)
    

    



else:

    unet = StandardUNet(1, 
              dim=3,
              base_filters=n_blowup_channels,
              zero_init=not non_invertible_unet_with_skip_connection,
              skip_connection=non_invertible_unet_with_skip_connection)
    
    module_list = [unet]
    
    # If a skip connection is used, the number of channels gets decreased to the number
    # of input channels again (otherwise the addition wouldn't be well-defined). If 
    # no skip connection is used, here we use a linear layer to change the number of
    # channels back.
    if not non_invertible_unet_with_skip_connection:
        collapse_layer = nn.Conv3d(2*n_blowup_channels, 1, 3, padding=1)
        #! For segmentation experiments:
        #! collapse_layer = nn.Conv3d(2*n_blowup_channels, n_classes, 1)
        module_list.append(collapse_layer)
        
    model = nn.Sequential(*module_list).to(device)
        

"""
END OF ARCHITECTURE DEFINITION
"""








"""
START OF LOSS AND OPTIMIZER DECLARATION
"""
loss_fn = torch.nn.MSELoss().to(device)
loss_fn.requires_grad = True

# Start gradient calculation
optimizer = torch.optim.Adam(list(model.parameters()), 
                             lr=initial_learning_rate, 
                             weight_decay=weight_decay_parameter)

# The following is a stepwise learning rate schedule, where 
def lr_schedule_fn(epoch):
    for i,e in enumerate(learning_rate_epochs):
        if epoch >= e:
            multiplier = learning_rate_multiplier[i]
    return multiplier

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_schedule_fn)
"""
END OF LOSS AND OPTIMIZER DECLARATION
"""




"""
START OF SUMMARY DECLARATION
"""

writer = SummaryWriter(
    log_dir=tensorboard_logdir,
    flush_secs=30)

"""
END OF SUMMARY DECLARATION
"""




"""
START OF LOADING SECTION
"""
# Create the folder for the saved models if it does not exist,
# otherwise the saver throws an error.
if saved_model_folder[-1] != '/':
    saved_model_folder += '/'
pathlib.Path(saved_model_folder + base_name).mkdir(parents=True,
                                                   exist_ok=True)

epoch = 0
if os.path.isfile(saved_model_path):
    checkpoint = torch.load(saved_model_path)
    checkpoint = torch.load(saved_model_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    print("Last checkpoint restored.")
"""
END OF LOADING SECTION
"""


do_validation = True
while epoch < num_epochs:
    print("Epoch", epoch)
    
    """
    BEGIN VALIDATION SECTION
    """
    if do_validation:
        SSIM_list = []
        PSNR_list = []
        val_loss_list = []
        loss_list = []

        for i in range(len(validation_data)):
            model.eval()
            with torch.no_grad():
                x, y = validation_data[i]
                validation_input = x.detach().cpu().numpy() 
                fx = model(x)
                loss = loss_fn(fx, y)
                val_loss_list.append(loss.detach().cpu().numpy())

                fx_ = fx.view(size, size, size).detach().cpu().numpy()
                y_ = y.view(size, size, size).detach().cpu().numpy()

                # might only be valid for batch size 1
                SSIM_list.append(calc_ssim(fx_,
                                           y_,
                                           window_size=11))

                PSNR_list.append(calc_psnr(fx_,
                                           y_,
                                           data_range=1))
                
                # No idea why, but if these are not manually deleted, Pytorch runs out
                # of memory after some validation runs
                del x, y, fx


        # Aggregate validation losses 
        mean_val_loss = np.mean(val_loss_list)
        mean_SSIM = np.mean(SSIM_list)
        mean_PSNR = np.mean(PSNR_list)

        # Add losses to tensorboard
        writer.add_scalar('validation/mean_loss', mean_val_loss, epoch)
        writer.add_scalar('validation/mean_SSIM', mean_SSIM, epoch)
        writer.add_scalar('validation/mean_PSNR', mean_PSNR, epoch)


        # Add images to tensorboard
        fx_image = fx_[size // 2, :, :]
        y_image = y_[size // 2, :, :]
        x_image = validation_input[0,0, size // 2, :, :]

        writer.add_image('image/output', 
                         fx_image.reshape(1,256,256), 
                         global_step=epoch) # Needs CHW
        writer.add_image('image/ground_truth', 
                         y_image.reshape(1,256,256), 
                         global_step=epoch)
        writer.add_image('image/input',
                         validation_input[0, 0, size // 2].reshape(1,256,256), 
                         global_step=epoch)
        
    """
    END VALIDATION SECTION
    """
    
    
    """
    BEGIN TRAINING SECTION
    """
    lr_scheduler.step()
    loss_list = []
    
    training_indices = list(range(len(train_data)))
    random.shuffle(training_indices)
    for i in training_indices:
        model.train()

        x, y = train_data[i]

        fx = model(x)
        loss = loss_fn(fx, y)
        
        loss.backward()
        loss_list.append(loss.detach().cpu().numpy())
        optimizer.step()
        optimizer.zero_grad()
        
    mean_loss = np.mean(loss_list)
    writer.add_scalar('train/mean_loss', mean_loss, epoch)
    
    epoch += 1
    
    # Save model
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()}, 
            saved_model_path)
    """
    END TRAINING SECTION
    """    

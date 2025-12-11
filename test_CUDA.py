import sys,torch,torchvision
print("torchvision version:",torchvision. __version__)
print("python版本:",sys.version)
print("torch version:",torch.__version__)
print("gpu可用:",torch.cuda.is_available())
print('CUDA version:',torch.version.cuda)
import transformers, sys, os
import torch
print("Transformers version actually running:", transformers.__version__)
print("Transformers file:", transformers.__file__)
print("Python executable:", sys.executable)

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


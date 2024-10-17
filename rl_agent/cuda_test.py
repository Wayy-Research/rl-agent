import torch
import tensorflow

import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available(): 
    print("Torch")
    print("CUDA is available!")
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available in torch.")

if tensorflow.test.is_gpu_available():
    print("TensorFlow")
    print(f"Num GPUs Available: {len(tensorflow.config.list_physical_devices('GPU'))}")
else:
    print("CUDA is not available in tensorflow.")

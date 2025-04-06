#!/bin/python3

import torch

try:
    torch.cuda.is_available()
    print("OK")
except Exception as e:
    print("ERROR: " + str(e))
    print("Check the following:")
    print("-> check: 'nvcc -V', if not installed: ")
    print("-> 'sudo apt install cuda'")
    print("-> 'sudo rmmod nvidia_uvm'")
    print("-> 'sudo modprobe nvidia_uvm'")

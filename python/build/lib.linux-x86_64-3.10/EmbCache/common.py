from collections import namedtuple
import torch

TensorMeta = namedtuple(
    "TensorMeta", ["shm_name", "fd", "ptr", "shm_size", "dtype", "shape"])

SHM_ALLOC_AMPL_TIMES: float = 1.2
MSG_DELIMETER: str = ";"
EMPTY_TENSOR_NAME = "EMPTY_TENSOR"
STOP_MSG = "STOP"


def tensor_initializer(shape, dtype):
    tensor = torch.zeros(shape, dtype=dtype)
    tensor.uniform_(-1, 1)
    return tensor


def zero_initializer(shape, dtype):
    tensor = torch.zeros(shape, dtype=dtype)
    return tensor


def str2dtype(input: str):
    if input == "int32":
        return torch.int32
    elif input == "int64":
        return torch.int64
    elif input == "float32":
        return torch.float32
    elif input == "float64":
        return torch.float64
    elif input == "bool":
        return torch.bool


def dtype_sizeof(input):
    if isinstance(input, str):
        if input == "int32":
            return 4
        elif input == "int64":
            return 8
        elif input == "float32":
            return 4
        elif input == "float64":
            return 8
        elif input == "bool":
            return 1
    else:
        if input == torch.int32:
            return 4
        elif input == torch.int64:
            return 8
        elif input == torch.float32:
            return 4
        elif input == torch.float64:
            return 8
        elif input == torch.bool:
            return 1

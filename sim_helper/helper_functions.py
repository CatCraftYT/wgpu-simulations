import numpy as np
import sys
from typing import Any

def get_args(arg_names: list[str]):
    args = sys.argv[1:]
    if len(args) != len(arg_names):
        print(f"Incorrect number of arguments provided. Required arguments are: {', '.join(arg_names)}")
        sys.exit(1)

    return {arg_names[i]: args[i] for i in range(len(args))}

def create_parameters(params: dict[str, Any], paramTypes: list[str]):
    paramArray = np.zeros((), dtype=list(zip(params.keys(), paramTypes)))

    for name,value in params.items():
        paramArray[name] = value
    

    return paramArray

def create_data_array(n_elements, dtype, data_function = None):
    data = np.zeros(n_elements, dtype=dtype)

    if data_function:
        data_function(data)
    
    return data
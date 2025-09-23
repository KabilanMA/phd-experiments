from typing import Union
from enum import Enum

class Storage(Enum):
    DENSE = 1
    CSR = 2
    CSC = 3
    COO = 4

class Tensor:
    def __init__(self, name: str, order: int, *dimensions: int):
        self.name = name
        self.order = order
        self.dimensions = dimensions
    
    def insert(self, indices: tuple, value: Union[float, int]) -> None:
        if (self.order != len(indices)):
            print(f"[Error] Invalid index : {indices}. {self.name} can only accept indices of order: {self.order}")

class Matrix(Tensor):
    def __init__(self, name: str, format:Storage , *dimensions):
        super.__init__(name, 2, dimensions)
        


class EinsumOp:
    def __init__(self, output, expr, inputs):
        self.output = output
        self.expr = expr
        self.inputs = inputs
        
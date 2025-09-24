from typing import Union, List, Tuple
from enum import Enum

class Storage(Enum):
    DENSE = 1
    CSR = 2
    CSC = 3
    COO = 4

class EINSUM_OP(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4

class Tensor:
    def __init__(self, name: str, order: int, dimensions: Tuple[int]):
        self.name = name
        self.order = order
        self.dimensions = dimensions
        self.values = []
        self.value_indices = []
    
    def __str__(self) -> str:
        return f"Tensor(name={self.name}, order={self.order}, dimensions={self.dimensions}, nnz={len(self.values)})"

    
    def insert(self, indices: tuple, value: Union[float, int]) -> None:
        if (self.order != len(indices)):
            print(f"[Error] Invalid index : {indices}. {self.name} can only accept indices of order: {self.order}")
            return
        self.values.append(value)
        self.value_indices.append(indices)
    
class Matrix(Tensor):
    def __init__(self, name: str, format:Storage , dimensions: Tuple[int]):
        super().__init__(name, 2, dimensions)
        self.format = format
    
    def __str__(self) -> str:
        return f"Matrix(name={self.name}, format={self.format}, dimensions={self.dimensions}, nnz={len(self.values)})"
    
    def insert(self, indices: tuple, value: Union[float, int]) -> None:
        super().insert(indices, value)

class EinsumOp:
    def __init__(self, output: Tensor, expr: str, operator: EINSUM_OP, inputs: Tuple[Tensor]):
        self.output: Tensor = output
        self.expr: str = expr
        self.inputs: Tuple[Tensor] = inputs
        self.operator: EINSUM_OP = operator
    
    def __str__(self) -> str:
        input_names = [t.name for t in self.inputs]
        return f"EinsumOp(expr='{self.expr}', operation='{self.operator}', inputs={input_names}, output={self.output.name})"
        
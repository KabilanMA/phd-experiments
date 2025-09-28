from typing import Union, Tuple
from enum import Enum

class Storage(Enum):
    DENSE = 1
    CSR = 2
    CSC = 3
    COO = 4

class TENSOR_OP(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4

# ----------------------------
# High-level IR (user-facing)
# ----------------------------
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
        if self.order != len(indices):
            print(f"[Error] Invalid index {indices}: {self.name} expects indices of order {self.order}")
            return
        self.values.append(value)
        self.value_indices.append(indices)

class Matrix(Tensor):
    def __init__(self, name: str, format: Storage, dimensions: Tuple[int]):
        super().__init__(name, 2, dimensions)
        self.format = format
    
    def __str__(self) -> str:
        return f"Matrix(name={self.name}, format={self.format}, dimensions={self.dimensions}, nnz={len(self.values)})"

class EinsumOp:
    def __init__(self, output: Tensor, expr: str, inputs: Tuple[Tensor], operator: TENSOR_OP = None):
        self.output: Tensor = output
        self.expr: str = expr
        self.inputs: Tuple[Tensor] = inputs
        self.operator: TENSOR_OP = operator
    
    def __str__(self) -> str:
        input_names = [t.name for t in self.inputs]
        return f"EinsumOp(expr='{self.expr}', operations='{'None' if self.operator == None else self.operator.name}', inputs={input_names}, output={self.output.name})"

# ----------------------------
# Low-level IR (for lowering/codegen)
# ----------------------------
class LoopVar:
    def __init__(self, name: str):
        self.name = name
    def __str__(self): return self.name

class LoopNest:
    def __init__(self, var: LoopVar, body: list):
        self.var = var
        self.body = body  # list of IR nodes
    def __str__(self): return f"Loop({self.var}) {{ {self.body} }}"

class Load:
    def __init__(self, tensor: Tensor, indices: Tuple[str]):
        self.tensor = tensor
        self.indices = indices
    def __str__(self): return f"Load({self.tensor.name}[{','.join(self.indices)}])"

class Store:
    def __init__(self, tensor: Tensor, indices: Tuple[str], value):
        self.tensor = tensor
        self.indices = indices
        self.value = value
    def __str__(self): return f"Store({self.tensor.name}[{','.join(self.indices)}] = {self.value})"

class BinaryOp:
    def __init__(self, op: TENSOR_OP, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
    def __str__(self): return f"({self.lhs} {self.op.name} {self.rhs})"

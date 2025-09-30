from __future__ import annotations
from typing import List, Tuple, Union, Iterable, Optional
from enum import Enum, auto
from dataclasses import dataclass, field


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


# ------------------------
# Base classes
# ------------------------
class Node:
    """Base node: common utilities (visitor/traversal hooks can be added)."""
    def children(self) -> Iterable["Node"]:
        return []
    def __repr__(self):
        return self.__str__()

class Stmt(Node):
    """Statement in the low-level IR (loop, store, assign, block, if)."""
    pass

class Expr(Node):
    """Expression (loads, binary ops, constants, variables)."""
    pass

# ------------------------
# Index / loop variable
# ------------------------
@dataclass
class IndexVar(Expr):
    name: str

    def __str__(self) -> str:
        return self.name

# ------------------------
# Expressions
# ------------------------
@dataclass
class Constant(Expr):
    value: Union[int, float]

    def __str__(self) -> str:
        if isinstance(self.value, int):
            return str(self.value)
        return f"{self.value}"

@dataclass
class ScalarVar(Expr):
    """A scalar variable (e.g., temporaries)."""
    name: str

    def __str__(self) -> str:
        return self.name

@dataclass
class Load(Expr):
    """Load from a tensor: e.g., A[i,j] where indices is a tuple of strings or IndexVar."""
    tensor_name: str
    indices: Tuple[Union[str, IndexVar], ...]  # each index can be IndexVar or literal

    def __str__(self) -> str:
        idxs = ",".join(str(i) for i in self.indices)
        return f"{self.tensor_name}[{idxs}]"

    def children(self):
        for i in self.indices:
            if isinstance(i, Node):
                yield i

@dataclass
class BinaryOp(Expr):
    op: TENSOR_OP
    lhs: Expr
    rhs: Expr

    def __str__(self) -> str:
        return f"({self.lhs} {self.op.value} {self.rhs})"

    def children(self):
        yield self.lhs
        yield self.rhs

# ------------------------
# Statements
# ------------------------
@dataclass
class Assign(Stmt):
    """Simple assignment: target = value"""
    target: Load                # left-hand side must be a Load (tensor element) or ScalarVar via wrapper
    value: Expr

    def __str__(self) -> str:
        return f"{self.target} = {self.value};"

    def children(self):
        yield self.target
        yield self.value

@dataclass
class Update(Stmt):
    """In-place update: target op= value, e.g., C[i,k] += expr"""
    target: Load
    op: TENSOR_OP
    value: Expr

    def __str__(self) -> str:
        return f"{self.target} {self.op.value}= {self.value};"

    def children(self):
        yield self.target
        yield self.value

@dataclass
class Block(Stmt):
    """Sequence of statements"""
    body: List[Stmt] = field(default_factory=list)

    def __str__(self) -> str:
        inner = "\n".join(str(s) for s in self.body)
        return inner

    def children(self):
        for s in self.body:
            yield s

@dataclass
class If(Stmt):
    cond: Expr
    then_branch: Block
    else_branch: Optional[Block] = None

    def __str__(self) -> str:
        s = f"if ({self.cond}) {{\n{indent(str(self.then_branch))}\n}}"
        if self.else_branch:
            s += f" else {{\n{indent(str(self.else_branch))}\n}}"
        return s

    def children(self):
        yield self.cond
        yield self.then_branch
        if self.else_branch:
            yield self.else_branch

# ------------------------
# Loop representation
# ------------------------
@dataclass
class ForLoop(Stmt):
    """Classic for-loop over an IndexVar.

    start and end are Expr (commonly Constant or ScalarVar).
    loop_var is IndexVar used in the body.
    """
    loop_var: IndexVar
    start: Expr      # inclusive start (usually Constant(0))
    end: Expr        # exclusive end
    body: Block
    is_reduction: bool = False  # hint: this loop is a reduction axis

    def __str__(self) -> str:
        return f"for (int {self.loop_var} = {self.start}; {self.loop_var} < {self.end}; ++{self.loop_var}) {{\n{indent(str(self.body))}\n}}"

    def children(self):
        yield self.loop_var
        yield self.start
        yield self.end
        yield self.body

# ------------------------
# Helpers
# ------------------------
def indent(s: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in s.splitlines())

# ------------------------
# Utility: small builder for a typical einsum lowering pattern
# ------------------------
def build_matmul_loop(A_name: str, B_name: str, C_name: str,
                      i: IndexVar, j: IndexVar, k: IndexVar,
                      I: Expr, J: Expr, K: Expr) -> ForLoop:
    """
    Build:
      for i in range(I):
        for k in range(K):
          C[i,k] = 0;
          for j in range(J):
            C[i,k] += A[i,j] * B[j,k];
    Returns top-level loop (over i).
    """
    # inner update: C[i,k] += A[i,j] * B[j,k];
    load_A = Load(A_name, (i, j))
    load_B = Load(B_name, (j, k))
    mul = BinaryOp(TENSOR_OP.MUL, load_A, load_B)
    load_C = Load(C_name, (i, k))
    update = Update(load_C, TENSOR_OP.ADD, mul)

    # inner j loop
    j_body = Block([update])
    j_loop = ForLoop(loop_var=j, start=Constant(0), end=J, body=j_body, is_reduction=True)

    # init C[i,k] = 0;
    init = Assign(load_C, Constant(0))
    k_body = Block([init, j_loop])
    k_loop = ForLoop(loop_var=k, start=Constant(0), end=K, body=k_body)

    i_body = Block([k_loop])
    i_loop = ForLoop(loop_var=i, start=Constant(0), end=I, body=i_body)
    return i_loop

# ------------------------
# Example usage (for quick testing)
# ------------------------
if __name__ == "__main__":
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    I = ScalarVar("M")
    J = ScalarVar("N")
    K = ScalarVar("P")

    top = build_matmul_loop("A", "B", "C", i, j, k, I, J, K)
    print(top)


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
        self.name: str = name

    def __str__(self): return self.name

class LoopNest:
    def __init__(self, var: LoopVar, body: list, maximum_limit: str):
        self.var: LoopVar = var
        self.body: list = body  # list of IR nodes
        self.maximum_limit: str = maximum_limit

    def __str__(self): return f"Loop({self.var}) {{ {self.body} }}"

class Load:
    def __init__(self, tensor: Tensor, index: Tuple[str]):
        self.tensor = tensor
        self.index = index

    def __str__(self): return f"Load({self.tensor.name}[{','.join(self.index)}])"

class IfStmt:
    def __init__(self, tensor: Tensor, index: Tuple[str]):
        self.tensor = tensor
        self.index = index

# class Store:
#     def __init__(self, tensor: Tensor, indices: Tuple[str], value):
#         self.tensor = tensor
#         self.indices = indices
#         self.value = value

#     def __str__(self): return f"Store({self.tensor.name}[{','.join(self.indices)}] = {self.value})"

class Store:
    def __init__(self, tensor: Tensor, index: str, value):
        self.tensor = tensor
        self.index = index
        self.value = value

    def __str__(self): return f"Store({self.tensor.name}[{','.join(self.index)}] = {self.value})"


@dataclass
class Expr:
    pass

@dataclass
class Var(Expr):
    name: str

@dataclass
class Int(Expr):
    value: int

@dataclass
class Double(Expr):
    value: float

@dataclass
class String(Expr):
    value: str

@dataclass
class BinaryOp:
    op: TENSOR_OP
    lhs: Expr
    rhs: Expr

# --- Expressions ---
@dataclass
class CExpr:
    pass

@dataclass
class CVar(CExpr):
    name: str

@dataclass
class CInt(CExpr):
    value: int

@dataclass
class CDouble(CExpr):
    value: float

@dataclass
class CString(CExpr):
    value: str

@dataclass
class CBinary(CExpr):
    op: str
    lhs: CExpr
    rhs: CExpr

@dataclass
class CCall(CExpr):
    func: str
    args: List[CExpr]

@dataclass
class CCast(CExpr):
    ctype: str
    expr: CExpr

@dataclass
class CMember(CExpr):
    expr: CExpr       # e.g. CVar("B")
    member: str       # "rows"
    deref: bool = True # use -> if True else .

@dataclass
class CIndex(CExpr):
    array: CExpr      # CVar or CMember
    index: CExpr      # index expression

# --- Statements ---
@dataclass
class CStmt:
    pass

@dataclass
class CDecl(CStmt):
    ctype: str
    name: str
    init: Optional[CExpr] = None

@dataclass
class CAssign(CStmt):
    lhs: CExpr
    rhs: CExpr

@dataclass
class CExprStmt(CStmt):
    expr: CExpr

@dataclass
class CFor(CStmt):
    init: CStmt        # usually CDecl or CAssign
    cond: CExpr
    step: CStmt        # usually CExprStmt with assignment/increment
    body: List[CStmt]

@dataclass
class CIf(CStmt):
    cond: CExpr
    then_body: List[CStmt]
    else_body: Optional[List[CStmt]] = None

@dataclass
class CReturn(CStmt):
    expr: Optional[CExpr] = None

# --- Top-level ---
@dataclass
class Include:
    header: str

@dataclass
class StructField:
    ctype: str
    name: str

@dataclass
class StructDecl:
    name: str
    fields: List[StructField]

@dataclass
class FunctionDecl:
    ret_type: str
    name: str
    params: List[Tuple[str, str]]   # [(ctype, name), ...]
    body: List[CStmt]


# ------------------------
# Example usage (for quick testing)
# ------------------------
if __name__ == "__main__":
    includes = [Include("<stdio.h>"), Include("<stdlib.h>"), Include("<time.h>"), Include("<math.h>"), Include("<stdarg.h>")]

    csr_fields = [
        StructField("int", "rows"),
        StructField("int", "cols"),
        StructField("int", "nnz"),
        StructField("int *", "row_ptr"),
        StructField("int *", "col_ind"),
        StructField("double", "val"),
    ]
    csr_struct = StructDecl("CSRMatrix", csr_fields)

    # Function signature: CSRMatrix multipleCSR(const CSRMatrix *B, const CSRMatrix *C)
    params = [("const CSRMatrix *", "B"), ("const CSRMatrix *", "C")]

    # Body statements (sketching main ones)
    body = []

    # int n = B->rows;
    body.append(CDecl("int", "n", CMember(CVar("B"), "rows", deref=True)))
    #int m = C->cols;
    body.append(CDecl("int", "n", CMember(CVar("C"), "rows", deref=True)))

    # int * row_ptr = calloc(n+1, sizeof(int));
    body.append(CDecl("int *", "row_ptr", CCall("calloc", [CBinary("+", CVar("n"), CInt(1)), CCall("sizeof", [CVar("int")])])))

    # int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    cond_cap = CBinary(">", CMember(CVar("B"), "nnz", deref=True), CMember(CVar("C"), "nnz", deref=True))
    cap_expr = CCall("(", [cond_cap])

    # Main Loop Construct
    # inner if: if (C->col_ind[k] == i) { col_ind[nnz] = j; val[nnz] = B->val[p] * C->val[k]; nnz++; }
    if_cond = CBinary("==", CIndex(CMember(CVar("C"), "col_ind", deref=True), CVar("k")), CVar("i"))
    inner_then = [
        CAssign(CIndex(CVar("col_ind"), CVar("nnz")), CVar("j")),
        CAssign(CIndex(CVar("val"), CVar("nnz")),
            CBinary("*", CIndex(CMember(CVar("B"), "val", deref=True), CVar("p")),
                          CIndex(CMember(CVar("C"), "val", deref=True), CVar("k")))),
        CExprStmt(CCall("++", [CVar("nnz")]))  # emit as "nnz++;"
    ]
    inner_if = CIf(if_cond, inner_then)

    # build inner-most loop: for (int k = C->row_ptr[j]; k < C->row_ptr[j+1]; k++) { inner_if }
    k_init = CDecl("int", "k", CMember(CMember(CVar("C"), "row_ptr", deref=True), "j"))  # rough; better to emit CMember[CIndex] but simplified
    k_cond = CBinary("<", CVar("k"), CMember(CMember(CVar("C"), "row_ptr", deref=True), "j+1")) 
    k_step = CExprStmt(CCall("++", [CVar("k")]))
    k_loop = CFor(k_init, k_cond, k_step, [inner_if])

    body.append(CExprStmt(CCall("/* final construct CSRMatrix A init */", [])))
    body.append(CReturn(CVar("A")))

    fn = FunctionDecl("CSRMatrix", "multiplyCSR", params, body)

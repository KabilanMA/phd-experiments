
from __future__ import annotations
from typing import List, Tuple, Union, Iterable, Optional
from enum import Enum, auto
from dataclasses import dataclass, field

class StorageDataType(Enum):
    INT = 0
    FLOAT = 1
    DOUBLE = 2

# ------------------ Expressions ------------------
class BaseExpr:
    def emit(self) -> str:
        raise NotImplementedError("emit() not implemented in BaseExpr")

class NumLiteral(BaseExpr):
    def __init__(self, value: Union[int, float]):
        self.value = value
    
    def emit(self) -> str:
        return str(self.value)

class VarRef(BaseExpr):
    def __init__(self, name: str):
        self.name = name
    
    def emit(self):
        return self.name

class BinaryExpr(BaseExpr):
    def __init__(self, left: BaseExpr, op: str, right: BaseExpr):
        self.left = left
        self.op = op
        self.right = right
    
    def emit(self) -> str:
        return f"({self.left.emit()} {self.op} {self.right.emit()})"

class ArrayAccess(BaseExpr):
    def __init__(self, array_name: StoreList, index: BaseExpr):
        self.array_name = array_name

# ------------------ Declarations ------------------

class VarDecl(BaseExpr):
    def __init__(self, name: str, type_: StorageDataType):
        self.name = name
        self.type = type_
    
    def emit(self, initial_val: Union[int, float] = None) -> str:
        if (initial_val is not None):
            return f"{self.type.name.lower()} {self.name} = {initial_val};"
        else:
            return f"{self.type.name.lower()} {self.name};"

class StoreList:
    def __init__(self, name: str, size: int = 0, data_type: StorageDataType = StorageDataType.INT):
        self.name = name
        self.size = size
        self.storage_data_type = data_type

    def __str__(self):
        return f"{self.name} with size {self.size}"
    
    def emit_decl(self):
        if self.storage_data_type == StorageDataType.INT:
            return f"int *{self.name} = malloc({self.size} * sizeof(int));"
        elif self.storage_data_type == StorageDataType.FLOAT:
            return f"float *{self.name} = malloc({self.size} * sizeof(float));"
        elif self.storage_data_type == StorageDataType.DOUBLE:
            return f"double *{self.name} = malloc({self.size} * sizeof(double));"
    
    def emit_free(self):
        return f"free({self.name});"
    
    def emit_store_val(self, index: str):
        return f"{self.name}[{index}]"
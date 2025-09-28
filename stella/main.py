from dsl.api import tensor_operation
from ir.lowering import lower_to_loops
from codegen.c_codegen import emit_c
from ir.ir import Tensor

if __name__ == "__main__":
    A = Tensor("A", 2, (3,3))
    B = Tensor("B", 2, (3,3))

    # DSL → H-IR
    op = tensor_operation("ij,ji->ij", A, B)
    print(op)

    # H-IR → L-IR
    loop_ir = lower_to_loops(op)
    print(loop_ir)

    # L-IR → C code
    c_code = emit_c(loop_ir)
    print("Generated C code:\n", c_code)

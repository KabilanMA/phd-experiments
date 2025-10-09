# from dsl.api import einsum_op
# from stella import stella
# from ir.lowering import lower_to_loops
# from codegen.c_codegen import emit_c
# from ir.ir import Tensor

import stella

if __name__ == "__main__":
    A = stella.Tensor("A", 2, (3,3))
    B = stella.Tensor("B", 2, (3,3))
    # DSL → H-IR
    op = stella.einsum_op("ij,ji->ij", A, B, operator="*")
    # print(op)

    # H-IR → L-IR
    loop_ir = stella.lower_to_loops(op)
    print(loop_ir)

    # # L-IR → C code
    c_code = stella.emit_c(loop_ir)
    print("Generated C code:\n", c_code)

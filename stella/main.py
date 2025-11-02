# from dsl.api import einsum_op
# from stella import stella
# from ir.lowering import lower_to_loops
# from codegen.c_codegen import emit_c
# from ir.ir import Tensor
from stella.ir.ir import Storage

import stella

if __name__ == "__main__":
    A = stella.Tensor("A", [3,6], [Storage.DENSE, Storage.SPARSE])
    B = stella.Tensor("B", [3,6], [Storage.DENSE, Storage.SPARSE])
    C = stella.Tensor("C", [6,3], [Storage.DENSE, Storage.SPARSE])
    # DSL → H-IR
    op = stella.tensor_op("ij,ji->ij", B, C, A, operator="*")

    tensor_info = stella.extract_info(op)

    stella.emit_c(op)
    # print(op)

    # # H-IR → L-IR
    # loop_ir = stella.lower_to_loops(op)
    # print(loop_ir)

    # # # L-IR → C code
    # c_code = stella.emit_c(loop_ir)
    # print("Generated C code:\n", c_code)

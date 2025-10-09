import sys
from stella.ir.ir import Tensor, EinsumOp, TENSOR_OP
from stella.dsl.parser import sanity_check_equation, find_output_order, find_output_dimension, find_operator

def einsum_op(equation: str, *inputs: Tensor, operator: str = "*") -> EinsumOp:
    """
    DSL entry point for defining Einstein operations. This can also support other 
    general tensor operations like addition, subtraction and divison too.

    Example:
        A = einsum_op("ij,jk->ik", B, C)
    """
    operator = operator.strip()
    if operator not in ["+", "-", "*", "/"]:
        print(f"[Error] Invalid operator '{operator}': only [+, -, *, /] supported")
        sys.exit(1)

    if not sanity_check_equation(equation, len(inputs)):
        sys.exit(1)
    
    print("[DSL] Sanity check completed.")

    output_order = find_output_order(equation)
    output_dimension = find_output_dimension(equation, inputs)

    # Build High-level IR
    output_tensor = Tensor("A", output_order, output_dimension)
    op = EinsumOp(output_tensor, equation, inputs, find_operator(operator))
    
    return op
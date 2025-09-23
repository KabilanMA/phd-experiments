import sys
from .ir import Tensor, EinsumOp

def sanity_check_equation(equation: str, tensor_count: int) -> bool:
    if "->" not in equation:
        print(f"[Error] Invalid einsum equation '{equation}': missing '->'.")
        return False
    lhs, rhs = equation.split("->")
    input_specs = lhs.split(",")
    if (len(input_specs) != tensor_count):
        print(f"[Erorr] Invalid tensor count '{equation}': mismatch between input tensor count")
        return False
    return True

def einsum(equation: str, operator: str, *inputs) -> Tensor:
    """
    DSL entry point for defining Einstein operations.

    Example:
        A = einsum("ij,ji->ij", B, C)
    """
    operator = operator.strip()
    if not ((operator in ["+", "-", "*", "/"])):
        print(f"[Error] Invalid operator '{operator}: kgen only support [+, -, *, /]")
        sys.exit(1)
    if (not sanity_check_equation(equation, len(inputs))):
        sys.exit(1)
    print(f"Sanity check completed")
    
    # TODO: You can add validation later (check matching dims, valid symbols, etc.)
    # output = Tensor("out", None)  # shape inference can come later
    # return EinsumOp(equation, list(inputs), output)

if __name__ == "__main__":
    einsum("ij,ji->ij", "*", Tensor("b", 3), Tensor("c", 3))
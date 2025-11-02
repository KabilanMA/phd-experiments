import sys
from stella.ir.ir import Tensor, TensorOp, Reduction
from stella.dsl.parser import sanity_check_equation, find_output_order, build_tensor_axis

def tensor_op(equation: str, *tensors: Tensor, operator: str = "*", reduction_strategy: Reduction = Reduction.auto) -> TensorOp:
    """
    Example:
        A = tensor_op("ij,jk->ik", B, C, A)
    """
    operator = operator.strip()
    if operator not in ["+", "-", "*", "/"]:
        print(f"[Error] Invalid operator '{operator}': only [+, -, *, /] supported")
        sys.exit(1)

    if not sanity_check_equation(equation, len(tensors)-1):
        sys.exit(1)
    
    output_order_calculated = find_output_order(equation)
    if (output_order_calculated != tensors[-1].order):
        print(f"[Error] Output order mismatch in the equation : {equation} with tensor: {tensors[-1].name}")
        sys.exit(1)
    
    reduction_technique = reduction_strategy
    # if (reduction_technique == Reduction.auto):
    #     reduction_strategy(find_reduction_strategy(equation))

    # print("[DSL] Sanity check completed.")

    build_tensor_axis(list(tensors), equation)
    output_tensor = tensors[-1]
    # TODO: Should find the tensor op and reduction technique via the parser
    op = TensorOp(output_tensor, equation, tensors[:-1], reduction=Reduction.non)
    
    return op

def extract_info(tensorOp: TensorOp, smt_info_file: str) -> dict :
    return {}

# def einsum_op(equation: str, *inputs: Tensor, operator: str = "*") -> EinsumOp:
#     """
#     DSL entry point for defining Einstein operations. This can also support other 
#     general tensor operations like addition, subtraction and divison too.

#     Example:
#         A = einsum_op("ij,jk->ik", B, C)
#     """
#     operator = operator.strip()
#     if operator not in ["+", "-", "*", "/"]:
#         print(f"[Error] Invalid operator '{operator}': only [+, -, *, /] supported")
#         sys.exit(1)

#     if not sanity_check_equation(equation, len(inputs)):
#         sys.exit(1)
    
#     print("[DSL] Sanity check completed.")

#     output_order = find_output_order(equation)
#     output_dimension = find_output_dimension(equation, inputs)

#     # Build High-level IR
#     output_tensor = Tensor("A", output_order, output_dimension)
#     op = EinsumOp(output_tensor, equation, inputs, find_operator(operator))
    
#     return op
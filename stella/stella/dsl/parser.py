import sys
from stella.ir.ir import TENSOR_OP, Reduction, Tensor, Axis, TensorOp
from typing import List, Tuple, Union, Iterable, Optional



def sanity_check_equation(equation: str, tensor_count: int) -> bool:
    # print(f"Equation: {equation}\ntensor_count: {tensor_count}")
    if "->" not in equation:
        print(f"[Error] Invalid einsum equation '{equation}': missing '->'.")
        return False
    lhs, rhs = equation.split("->")
    input_specs = lhs.split(",")
    if (len(input_specs) != tensor_count):
        print(f"[Error] Invalid tensor count '{equation}': mismatch between input tensor count")
        return False
    for iteration in input_specs:
        for char in iteration:
            if not char.isalpha():
                print(f"[Error] Invalid input einsum equation: {equation} in {iteration} with {char}")
                return False
    return True

def find_output_order(equation: str) -> int:
    _, rhs = equation.split("->")
    rhs = rhs.strip()
    count = 0
    for num in rhs:
        if num.isalpha():
            count+=1
    return count

def _find_axis_names(equaion: str):
    input, output = equaion.split("->")
    input_tensor_axes = input.split(",")
    input_tensor_axes.append(output)
    return input_tensor_axes

def build_tensor_axis(tensors: List[Tensor], equation: str):
    axes_names = _find_axis_names(equation)
    if (len(axes_names) != len(tensors)):
        print(f"[Error] Incorrect tensor count with axis names at build_tensor_axis(tensors: List[Tensor], equation: str)")
        sys.exit(1)
    
    for i in range(len(tensors)):
        tensor = tensors[i]
        axes_str = axes_names[i]
        axes_list = list(axes_str)
        for j in range(len(axes_list)):
            axis_str = axes_list[j]
            axis = Axis(axis_str, tensor.storage_fmt[j], tensor.name, tensor.dimensions[j])
            tensor.axis.append(axis)

# TODO: Complete it properly
def find_reduction_strategy(equation: str) -> Reduction:
    _,rhs = equation.split("->")
    rhs = rhs.strip()

    return Reduction.auto


# def find_output_dimension(equation: str, inputs: Tuple[Tensor]) -> tuple:
#     if (not sanity_check_equation(equation, len(inputs))):
#         sys.exit(1)

#     lhs, rhs = equation.split("->")
#     input_specs = lhs.split(",")
#     dimension_locations = []
#     for rhs_index in rhs:
#         if rhs_index.isalpha():
#             for input_spec_ind in range(len(input_specs)):
#                 found_location = input_specs[input_spec_ind].find(rhs_index)
#                 if found_location<0:
#                     continue
#                 dimension_locations.append((input_spec_ind, found_location))
#                 break
#     if (len(rhs.strip()) != len(dimension_locations)):
#         print(f"[Error] Invalid iteration format in the equation and mismatch between the iteration space and the input matrix space")
#         sys.exit(1)
    
#     output_dimension = []
#     for dimension_location in dimension_locations:
#         output_dimension.append(inputs[dimension_location[0]].dimensions[dimension_location[1]])
    
#     return tuple(output_dimension)

# def find_operator(operator: str) -> TENSOR_OP:
#     possible_operators = {"+": TENSOR_OP.ADD, "-": TENSOR_OP.SUB, "*": TENSOR_OP.MUL, "/": TENSOR_OP.DIV}
#     try:
#         return possible_operators[operator.strip()]
#     except KeyError:
#         print(f"[Error] Invalid operator {operator}: only support [+, -, *, /] operators")
#         sys.exit(1)    
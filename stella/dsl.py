# import sys
# from .ir import Tensor, EinsumOp, Matrix, Storage, EINSUM_OP
# from typing import Tuple

# def sanity_check_equation(equation: str, tensor_count: int) -> bool:
#     # print(f"Equation: {equation}\ntensor_count: {tensor_count}")
#     if "->" not in equation:
#         print(f"[Error] Invalid einsum equation '{equation}': missing '->'.")
#         return False
#     lhs, rhs = equation.split("->")
#     input_specs = lhs.split(",")
#     if (len(input_specs) != tensor_count):
#         print(f"[Error] Invalid tensor count '{equation}': mismatch between input tensor count")
#         return False
#     for iteration in input_specs:
#         for char in iteration:
#             if not char.isalpha():
#                 print(f"[Error] Invalid input einsum equation: {equation} in {iteration} with {char}")
#                 return False
#     return True

# def find_output_order(equation: str) -> int:
#     _, rhs = equation.split("->")
#     rhs = rhs.strip()
#     count = 0
#     for num in rhs:
#         if num.isalpha():
#             count+=1
#     return count

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

    

# def einsum(equation: str, operator: str, *inputs: Tensor) -> Tensor:
#     """
#     DSL entry point for defining Einstein operations.

#     Example:
#         A = einsum("ij,ji->ij", B, C)
#     """
#     operator = operator.strip()
#     if not ((operator in ["+", "-", "*", "/"])):
#         print(f"[Error] Invalid operator '{operator}: kgen only support [+, -, *, /]")
#         sys.exit(1)
#     if (not sanity_check_equation(equation, len(inputs))):
#         sys.exit(1)
#     print(f"Sanity check completed")

#     output_order = find_output_order(equation)
#     output_dimension = find_output_dimension(equation, inputs)
#     # print(output_dimension)
#     EinsumOp(Tensor("A", output_order, output_dimension), equation, )
    
#     # TODO: You can add validation later (check matching dims, valid symbols, etc.)
#     # output = Tensor("out", None)  # shape inference can come later
#     # return EinsumOp(equation, list(inputs), output)

# if __name__ == "__main__":
#     B = Matrix("B", Storage.CSR, (5, 5))
#     C = Matrix("C", Storage.CSR, (5, 5))
#     einsum("ij,ji->ij", "*", B, C)
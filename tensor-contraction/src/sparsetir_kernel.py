import tvm
from tvm.script import tir as T
import torch as th
import numpy as np

m, n = 4, 4
# random dense arrays to simulate sparse CSR input
B_np = np.random.rand(m, n).astype("float32")
C_np = np.random.rand(n, m).astype("float32")

# Convert dense to CSR using scipy
import scipy.sparse as sp
B_csr = sp.csr_matrix(B_np)
C_csr = sp.csr_matrix(C_np)

# Extract CSR arrays
B_data = B_csr.data
B_indices = B_csr.indices
B_indptr = B_csr.indptr

C_data = C_csr.data
C_indices = C_csr.indices
C_indptr = C_csr.indptr

# Define TIR sparse kernel
@T.prim_func
def csr_elemwise_mul(
    B: T.handle,
    C: T.handle,
    A: T.handle,
    B_indptr: T.handle,
    B_indices: T.handle,
    C_indptr: T.handle,
    C_indices: T.handle,
    nnz_B: T.int32,
    nnz_C: T.int32,
):
    T.func_attr({"global_symbol": "csr_elemwise_mul", "tir.noalias": True, "sparse_tir_level": 2})
    
    I = T.sparse_variable(nnz_B, (m, n), (B_indptr, B_indices), "int32")
    J = T.sparse_variable(nnz_C, (n, m), (C_indptr, C_indices), "int32")

    B_buf = T.match_sparse_buffer(B, (I,), "float32")
    C_buf = T.match_sparse_buffer(C, (J,), "float32")
    A_buf = T.match_sparse_buffer(A, (I,), "float32")  # same sparsity as B

    with T.sp_iter([I, J], "SR", "csr_elemwise_mul") as [i, j]:
        with T.init():
            A_buf[i] = 0.0
        A_buf[i] = B_buf[i] * C_buf[j]

# Build module
mod = tvm.build(csr_elemwise_mul, target="llvm")

# Create TVM NDArrays
ctx = tvm.cpu()
B_tvm = tvm.nd.array(B_data, ctx)
C_tvm = tvm.nd.array(C_data, ctx)
A_tvm = tvm.nd.empty_like(B_tvm)
B_indptr_tvm = tvm.nd.array(B_indptr.astype("int32"), ctx)
B_indices_tvm = tvm.nd.array(B_indices.astype("int32"), ctx)
C_indptr_tvm = tvm.nd.array(C_indptr.astype("int32"), ctx)
C_indices_tvm = tvm.nd.array(C_indices.astype("int32"), ctx)

# Run
mod(B_tvm, C_tvm, A_tvm, B_indptr_tvm, B_indices_tvm, C_indptr_tvm, C_indices_tvm,
    B_data.size, C_data.size)

# Inspect result
print("B data:", B_tvm.asnumpy())
print("C data:", C_tvm.asnumpy())
print("A result:", A_tvm.asnumpy())

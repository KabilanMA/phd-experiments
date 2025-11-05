### Kernel 1 - A(i, j) = B(i, j) * C(j, i)

|   No | A   |   B   |   C   | Axis Order |Search Tensor |Done | Note |
|---|---|---|---|---|---|---|---|
| 1 | CSR | CSR | CSR | i-j | C | ✅ | None |
| 2 | CSR | CSR | CSC | i-j | C | ✅ | None |
| 3 | CSR | CSR | COO | i-j | C | ✅ | None (Don't run this, it will take a lot of time for denser or large dimension tensor) |
| 4 | CSR | CSC | CSR | Any |C | ❌ | B is iterating through j (column) first, therefore it's impossible to find the correct column_index value to store in A without computing the entire tensor operation |
| 5 | CSR | CSC | CSC | Any | C | ❌ | B is iterating through j (column) first, therefore it's impossible to find the correct column_index value to store in A without computing the entire tensor operation |
| 6 | CSR | CSC | COO | Any | C | ❌ | B is iterating through j (column) first, therefore it's impossible to find the correct column_index value to store in A without computing the entire tensor operation |
| 7 | CSR | COO | CSR | i-j/j-i | C | ✅ | Since B is in COO format, there is no specific order in i or j. This computation is only possible because COO format store the indices from the inner most dimension to outer most dimension, i.e., B's COO is stored in row-wise manner. |
| 8 | CSR | COO | CSC | i-j/j-i | C | ✅ | Since B is in COO format, there is no specific order in i or j. This computation is only possible because COO format store the indices from the inner most dimension to outer most dimension, i.e., B's COO is stored in row-wise manner. |
| 9 | CSR | COO | C00 | i-j/j-i | C | ✅ | None (Don't run this, it will take a lot of time for denser or large dimension tensor) |
| 10 | CSC | CSR | CSR | Any | C | ❌ | Cannot be done without expensive operations. Look at note below. |
| 11 | CSC | CSR | CSC | Any | C | ❌ | Cannot be done without expensive operations. Look at note below. |
| 12 | CSC | CSR | COO | Any | C | ❌ | Cannot be done without expensive operations. Look at note below. |
| 13 | CSC | CSC | CSR | j-i | C | ✅ | Since B is in CSC, to iterate through B we will have to first iterate j and then i |
| 14 | CSC | CSC | CSC | j-i | C | ✅ | Since B is in CSC, to iterate through B we will have to first iterate j and then i |
| 15 | CSC | CSC | COO | j-i | C | ✅ | Since B is in CSC, to iterate through B we will have to first iterate j and then i (Don't run this, it will take a lot of time for denser or large dimension tensor) |
| 16 | CSC | COO | CSR | j-i | C | ❌ | Our COO format is stored in row-wise manner as mentioned above. Therefore B can go iterate through row-wise only. Until we complete all the rows of B, element-wise multipled by C, we cannot create the column_ptr for result A in CSC format. |
| 17 | CSC | COO | CSC | j-i | C | ❌ | Our COO format is stored in row-wise manner as mentioned above. Therefore B can go iterate through row-wise only. Until we complete all the rows of B, element-wise multipled by C, we cannot create the column_ptr for result A in CSC format. |
| 18 | CSC | COO | COO | j-i | C | ❌ | Our COO format is stored in row-wise manner as mentioned above. Therefore B can go iterate through row-wise only. Until we complete all the rows of B, element-wise multipled by C, we cannot create the column_ptr for result A in CSC format. |
| 19 | CSR | CSR | CSR | j-i | C | ❌ | Even though A, B and C are in CSR, we cannot store the result of B(i,j) * C(j,i) in A(i,j) with loop permutation j-i. Only way is to compute the entire computation of the tensor and then store the value into A, which is very expensive. |
| 20 | CSR | CSR | CSC | j-i | C | ❌ | It's the same case No. 19. For j-i, A needs to be in CSC. |
| 21 | CSR | CSR | COO | j-i | C | ❌ | Same as No. 20 |
| 22 | CSR | CSC | CSR | j-i | C | ❌ | Same as No. 20 |
| 23 | CSR | COO | CSR | j-i | C | ❌ | Same as No. 20 |
| 24 | CSC | CSR | CSR | j-i | C | ✅ | None |
| 25 | CSC | CSR | CSC | j-i | C | ✅ | None |
| 26 | CSC | CSR | COO | j-i | C | ✅ | None (Don't run this, it will take a lot of time for denser or large dimension tensor)
| 27 | CSC | CSC | CSR | i-j | C | ❌ | Same as No. 20 |
| 28 | CSC | CSC | CSC | i-j | C | ❌ | Same as No. 20 |
| 29 | CSC | CSC | COO | i-j | C | ❌ | Same as No. 20 |
| 30 | CSC | COO | CSR | i-j | C | ❌ | Same as No. 16 |
| 31 | CSC | COO | CSC | i-j | C | ❌ | Same as No. 17 |
| 32 | CSC | COO | COO | i-j | C | ❌ | Same as No. 18 |
|---|---|---|---|---|---|---|---|
| 33 | CSR | CSR | CSR | j-i | B | ✅ | None |
| 34 | CSR | CSR | CSR | i-j | B | ✅ | None |
| 35 | CSR | CSR | CSC | j-i | B | ✅ | None |
| 36 | CSR | CSR | CSC | i-j | B | ❌ | A needs to be in CSC to store the results back in the compressed format directly. |
| 37 | CSR | CSR | COO | j-i | B | ❌ | Same as 36 |



__Note__

If `A` needs to be in CSR, `B` should be in CSR too (for i-j loop permutation).
If `B` needs to be in CSC, `C` should be in CSC too (for i-j loop permutation).
If not, those computations will be very expensive. Because to create the resultant tensor `A`, all computations needs to completed.

Unzipper 16, 17, 18 can be done if we change the COO format structure to store the data in the column-wise order, not row-wise order.

### Kernel 2 - A(i, j) = B(i, k) * C(k, j)

|   No | A   |   B   |   C   | Axis Order |Search Tensor |Done | Note |
|---|---|---|---|---|---|---|---|
| 1 | CSR | CSR | CSR | i-j | C | ✅ | None |
| 2 | CSR | CSR | CSC | i-j | C | ✅ | None |
| 3 | CSR | CSR | COO | i-j | C | ✅ | Don't run this because it will take too much time to iterate through C in COO format |
| 4 | CSR | CSC | CSR | i-j | C | ✅ | Don't run this because it will take too much time to iterate through B in row-wise, which is in column-wise CSC format |
| 5 | CSR | CSC | CSC | i-j | C | ✅ | Don't run this because it will take too much time to iterate through B in row-wise, which is in column-wise CSC format |
| 6 | CSR | CSC | COO | i-j | C |  | None |
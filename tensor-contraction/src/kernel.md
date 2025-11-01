|   No | A   |   B   |   C   | Axis Order |Search Tensor |Done | Note |
|---|---|---|---|---|---|---|---|
| 1 | CSR | CSR | CSR | i-j | C | ✅ | None |
| 2 | CSR | CSR | CSC | i-j | C | ✅ | None |
| 3 | CSR | CSR | COO | i-j | C | ✅ | None |
| 4 | CSR | CSC | CSR | Any |C | ❌ | B is iterating through j (column) first, therefore it's impossible to find the correct column_index value to store in A without computing the entire tensor operation |
| 5 | CSR | CSC | CSC | Any | C | ❌ | B is iterating through j (column) first, therefore it's impossible to find the correct column_index value to store in A without computing the entire tensor operation |
| 6 | CSR | CSC | COO | Any | C | ❌ | B is iterating through j (column) first, therefore it's impossible to find the correct column_index value to store in A without computing the entire tensor operation |
| 7 | CSR | COO | CSR | i-j/j-i | C | ✅ | Since B is in COO format, there is no specific order in i or j. This computation is only possible because COO format store the indices from the inner most dimension to outer most dimension, i.e., B's COO is stored in row-wise manner. |
| 8 | CSR | COO | CSC | i-j/j-i | C | ✅ | Since B is in COO format, there is no specific order in i or j. This computation is only possible because COO format store the indices from the inner most dimension to outer most dimension, i.e., B's COO is stored in row-wise manner. |
| 9 | CSR | COO | C00 | i-j/j-i | C | ✅ | None |
| 10 | CSC | CSR | CSR | Any | C | ❌ | Cannot be done without expensive operations. Look at note below. |
| 11 | CSC | CSR | CSC | Any | C | ❌ | Cannot be done without expensive operations. Look at note below. |
| 12 | CSC | CSR | COO | Any | C | ❌ | Cannot be done without expensive operations. Look at note below. |
| 13 | CSC | CSC | CSR | j-i | C | ✅ | Since B is in CSC, to iterate through B we will have to first iterate j and then i |
| 14 | CSC | CSC | CSC | j-i | C | ✅ | Since B is in CSC, to iterate through B we will have to first iterate j and then i |
| 15 | CSC | CSC | COO | j-i | C | ✅ | Since B is in CSC, to iterate through B we will have to first iterate j and then i |
| 16 | CSC | COO | CSR | Any | C | ❌ | Our COO format is stored in row-wise manner as mentioned above. Therefore B can go iterate through row-wise only. Until we complete all the rows of B, element-wise multipled by C, we cannot create the column_ptr for result A in CSC format. |
| 17 | CSC | COO | CSC | Any | C | ❌ | Our COO format is stored in row-wise manner as mentioned above. Therefore B can go iterate through row-wise only. Until we complete all the rows of B, element-wise multipled by C, we cannot create the column_ptr for result A in CSC format. |
| 18 | CSC | COO | COO | Any | C | ❌ | Our COO format is stored in row-wise manner as mentioned above. Therefore B can go iterate through row-wise only. Until we complete all the rows of B, element-wise multipled by C, we cannot create the column_ptr for result A in CSC format. |

__Note__

If `A` needs to be in CSR, `B` should be in CSR too.

If `B` needs to be in CSC, `C` should be in CSC too.

If not, those computations will be very expensive. Because to create the resultant tensor `A`, all computations needs to completed.

Unzipper 16, 17, 18 can be done if we change the COO format structure to store the data in the column-wise order, not row-wise order.
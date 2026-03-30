import numpy as np

def augmenting_path(cost, u, v, path, row4col, shortest_path_costs, i, SR, SC):
    nc = cost.shape[1]
    min_val = 0
    num_remaining = nc
    remaining = np.arange(nc - 1, -1, -1)

    SR[:] = False
    SC[:] = False
    shortest_path_costs[:] = np.inf

    sink = -1
    SR[i] = True
    while sink == -1:
        index = -1
        lowest = np.inf
        
        remaining_range=np.arange(remaining.shape[0])
        j = remaining[remaining_range]
        r = min_val + cost[i, remaining] - u[i] - v[remaining]


        path[remaining]=i
        update_mask = r < shortest_path_costs[remaining]
        shortest_path_costs[remaining[update_mask]] = r[update_mask]

        
        lowest = np.min(shortest_path_costs[remaining])
        for it in remaining_range:
            j = remaining[it]
            if (shortest_path_costs[j] < lowest or 
            (shortest_path_costs[j] == lowest and row4col[j] == -1)):
                lowest = shortest_path_costs[j]
                index = it
        
        print(index)

        # lowest_index=np.where((shortest_path_costs[remaining]==lowest)&(row4col==-1))[0]
        # index = lowest_index
        

        # for it in range(num_remaining):
        # j = remaining[it]
            

        #     if r < shortest_path_costs[j]:
        #         path[j] = i
        #         shortest_path_costs[j] = r

        #     if (shortest_path_costs[j] < lowest or 
        #        (shortest_path_costs[j] == lowest and row4col[j] == -1)):
        #         lowest = shortest_path_costs[j]
        #         index = it

        min_val = lowest
        j = remaining[index]
        if min_val == np.inf:
            return -1, min_val

        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        SC[j] = True
        mask=np.arange(remaining.shape[0])!=index
        
        remaining = remaining[mask]
        # remaining = np.delete(remaining, index)
        
        
        num_remaining -= 1

    return sink, min_val,path,shortest_path_costs, SR, SC



# def augmenting_path(cost, u, v, path, row4col, shortest_path_costs, i, SR, SC):
#     nc = cost.shape[1]
#     path.fill(nc)  # Ensure all elements in path are initially set to nc
#     min_val = 0
#     SR[:] = False
#     SC[:] = False
#     shortest_path_costs[:] = np.inf
#     remaining = np.arange(nc)
#     cur_row=i
#     sink = -1
#     SR[i] = True
#     while sink==-1:
        

#         # Vectorized computation of reduced cost
#         reduced_costs = min_val + cost[i, remaining] - u[i] - v[remaining]
        
#         # Update shortest_path_costs and path arrays
#         update_mask = reduced_costs < shortest_path_costs[remaining]
#         shortest_path_costs[remaining[update_mask]] = reduced_costs[update_mask]
#         path[remaining[update_mask]] = i
        
#         # Find the minimum value and index among the updated shortest_path_costs
#         #
#         min_index = np.argmin(shortest_path_costs[remaining])
#         j = remaining[min_index]
#         min_val = shortest_path_costs[remaining[min_index]]
        
        
        

#         if min_val == np.inf:
#             return -1, min_val, path, shortest_path_costs, SR, SC

#         # Check if the current column `j` is unmatched
#         if row4col[j] == -1:
#             sink = j
#         else:
#             i = row4col[j]
        

        
#         remaining = remaining[np.arange(remaining.shape[0])!=min_index]
        

#     return sink, min_val, path, shortest_path_costs, SR, SC




def solve(input_cost):
    nr, nc = input_cost.shape
    cost = input_cost - np.min(input_cost)

    u = np.zeros(nr)
    v = np.zeros(nc)
    shortest_path_costs = np.full(nc, np.inf)
    path = np.full(nc, -1, dtype=int)
    col4row = np.full(nr, -1, dtype=int)
    row4col = np.full(nc, -1, dtype=int)
    SR = np.zeros(nr, dtype=bool)
    SC = np.zeros(nc, dtype=bool)

    row_range=np.arange(nr)

    for cur_row in row_range:
        min_val = 0
        sink, min_val,path,shortest_path_costs,SR, SC = augmenting_path(cost, u, v, path, row4col, shortest_path_costs, cur_row, SR, SC)

        if sink < 0:
            return -1

        u[cur_row] += min_val

        u[SR & (row_range != cur_row)] += min_val - shortest_path_costs[col4row[SR & (row_range != cur_row)]]

        # Update v values for columns where SC is True
        v[SC] -= min_val - shortest_path_costs[SC]  

        j = sink
        while True:
            i = path[j]
            # print('col4row[i]',col4row[i])
            row4col[j] = i
            temp_j=j
            j  = col4row[i]
            col4row[i] = temp_j
            # print('i',i)
            # print('path',path)
            # print('j',j)
            # print('col4row[i] dopo',col4row[i])
            # input()
            if i == cur_row:
                break

    return col4row


def hungarian(Cost:np.ndarray):
    J,W=Cost.shape
    assert J<=W,'wrong matrix shape'

    wo




# import numpy as np

# def augmenting_path(cost, u, v, path, row4col, shortest_path_costs, i, SR, SC):
#     nc = cost.shape[1]
#     path.fill(nc)  # Initialize all elements in path to nc
#     SR[:] = False
#     SC[:] = False
#     shortest_path_costs[:] = np.inf

#     # Mark the starting row as visited
#     SR[i] = True

#     # Compute reduced costs for all columns in a single pass
#     reduced_costs = cost[i, :] - u[i] - v
#     update_mask = reduced_costs < shortest_path_costs
#     shortest_path_costs[update_mask] = reduced_costs[update_mask]
#     path[update_mask] = i

#     # Identify potential sink columns that are unmatched
#     unmatched = (row4col == -1)
#     candidates = shortest_path_costs[unmatched]
#     unmatched_indices = np.arange(nc)[unmatched]

#     # If no unmatched columns are reachable, return no sink
#     if len(candidates) == 0 or np.all(candidates == np.inf):
#         return -1, np.inf

#     # Select the unmatched column with the minimum cost as the sink
#     min_index = np.argmin(candidates)
#     sink = unmatched_indices[min_index]
#     min_val = candidates[min_index]

#     # Update SC to mark the selected column
#     SC[sink] = True

#     return sink, min_val

# def solve(input_cost):
#     nr, nc = input_cost.shape
#     cost = input_cost - np.min(input_cost)  # Normalize cost matrix

#     u = np.zeros(nr)
#     v = np.zeros(nc)
#     shortest_path_costs = np.full(nc, np.inf)
#     path = np.full(nc, -1, dtype=int)
#     col4row = np.full(nr, -1, dtype=int)
#     row4col = np.full(nc, -1, dtype=int)
#     SR = np.zeros(nr, dtype=bool)
#     SC = np.zeros(nc, dtype=bool)

#     for cur_row in range(nr):
#         sink, min_val = augmenting_path(
#             cost, u, v, path, row4col, shortest_path_costs, cur_row, SR, SC
#         )

#         if sink < 0:
#             return -1

#         # Update potentials vector u and v
#         u[SR] += min_val - shortest_path_costs[col4row[SR]]
#         v[SC] -= min_val - shortest_path_costs[SC]

#         # Path augmentation from sink to current row
#         j = sink
#         while True:
#             i = path[j]
#             row4col[j], col4row[i] = i, j
#             j = col4row[i]
#             if i == cur_row:
#                 break

#     return col4row

def solve_rectangular_linear_sum_assignment(input_cost):
    col4row = solve(input_cost)
    not_finished=col4row is None 
    return None if not_finished else col4row

import numpy as np

def calculate_assignment(cost_matrix):
    # Ensure the input is a NumPy array of type double and 2D
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    cast_matrix = cost_matrix
    if cost_matrix.ndim != 2:
        raise TypeError("Invalid cost matrix object, must be a 2D array")

    # Get dimensions of the matrix
    num_rows, num_cols = cost_matrix.shape

    # Check for NaN or -inf entries
    if np.isnan(cost_matrix).any() or np.isneginf(cost_matrix).any():
        raise ValueError("Matrix contains invalid numeric entries")

    # Create an array `a` with indices of rows (0, 1, ..., num_rows-1)
    a = np.arange(num_rows, dtype=np.int64)

    # Call the solve function to get the column assignment for each row
    try:
        b = solve_rectangular_linear_sum_assignment(cost_matrix)
    except Exception:
        raise ValueError("Cost matrix is infeasible")

    # Package results in a tuple similar to Py_BuildValue("OO", a, b)
    return a, np.array(b, dtype=np.int64)

# Example usage
input_cost = (np.random.rand(20,20)*25).astype(int)

print(input_cost)

 

import timeit
func=lambda :calculate_assignment(input_cost)
tempo=timeit.timeit(func,number=100)
print("Column assignment for each row:", func())

print('timing for one execution ', tempo)
column,line=func()
cost_total=np.sum(input_cost[column,line])


from scipy.optimize import linear_sum_assignment

func2=lambda:linear_sum_assignment(input_cost)
column,line=func2()
cost_total2=np.sum(input_cost[column,line])
tempo2=timeit.timeit(func2,number=100)
print('timing for one execution scipi ', tempo2)
print("Column assignment for each row:", func2())

print(f'numpy cost total {cost_total} scipi cost total {cost_total2}')






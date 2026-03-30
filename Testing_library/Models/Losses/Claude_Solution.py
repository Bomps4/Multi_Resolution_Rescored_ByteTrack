import numpy as np
from scipy.optimize import linear_sum_assignment

def augmenting_path(cost, u, v, path, row4col, shortest_path_costs, i, SR, SC):
    nc = cost.shape[1]
    path.fill(nc)  # Ensure all elements in path are initially set to nc
    min_val = 0
    SR[:] = False
    SC[:] = False
    shortest_path_costs[:] = np.inf
    remaining = np.arange(nc)

    sink = -1
    while sink == -1:
        SR[i] = True

        # Vectorized computation of reduced cost
        reduced_costs = min_val + cost[i, remaining] - u[i] - v[remaining]
        
        # Update shortest_path_costs and path arrays
        update_mask = reduced_costs < shortest_path_costs[remaining]
        shortest_path_costs[remaining[update_mask]] = reduced_costs[update_mask]
        path[remaining[update_mask]] = i
        
        # Find the minimum value and index among the updated shortest_path_costs
        min_index = np.argmin(shortest_path_costs[remaining])
        min_val = shortest_path_costs[remaining[min_index]]
        j = remaining[min_index]

        if min_val == np.inf:
            return -1, min_val, path, shortest_path_costs, SR, SC

        # Check if the current column `j` is unmatched
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        SC[j] = True
        remaining = np.delete(remaining, min_index)
        if remaining.size==0:
            break

    return sink, min_val, path, shortest_path_costs, SR, SC



def solve(cost, maximize=False):
    if cost.size == 0:
        return np.array([]), np.array([])
    
    nr, nc = cost.shape
    transpose = nc < nr
    
    if transpose:
        cost = cost.T
        nr, nc = nc, nr
    
    if maximize:
        cost = -cost
    
    u = np.zeros(nr)
    v = np.zeros(nc)
    path = np.full(nc, -1)
    row4col = np.full(nc, -1)
    SR = np.zeros(nr, dtype=bool)
    SC = np.zeros(nc, dtype=bool)
    SR.fill(False)
    SC.fill(False)
    for cur_row in range(nr):
        
        sink, min_val, path,row4col, SR, SC = augmenting_path(cost, u, v, path, row4col, cur_row, SR, SC)
        
        if sink < 0:
            raise ValueError("Linear assignment problem is infeasible.")
        
        u[cur_row] += min_val
        u[SR] += min_val - cost[SR, row4col[SR]] + u[SR] + v[row4col[SR]]
        v[SC] -= min_val - cost[path[SC], SC] + u[path[SC]] + v[SC]
        
        while True:
            i = path[sink]
            row4col[sink], sink = i, row4col[i]
            if i == cur_row:
                break
    
    if transpose:
        return np.argsort(row4col), np.arange(nr)
    else:
        return np.arange(nr), row4col

# Example usage
cost_matrix = np.random.rand(30, 30)
row_ind, col_ind = solve(cost_matrix)

# Verify the result using scipy's linear_sum_assignment
scipy_row_ind, scipy_col_ind = linear_sum_assignment(cost_matrix)

print('number one ',row_ind, col_ind)

print('number two',scipy_row_ind, scipy_col_ind)


assert np.allclose(row_ind, scipy_row_ind)
assert np.allclose(col_ind, scipy_col_ind)

print("Optimization successful!")
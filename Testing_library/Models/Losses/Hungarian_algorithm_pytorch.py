import torch 

def augmenting_path(cost, u, v, path, row4col, shortest_path_costs, i, SR, SC):
    # device=cost.device
    
    # min_val = 0
    

    # SR[:] = False
    # SC[:] = False
    # shortest_path_costs[:] = torch.inf
    # cost_clone=cost.clone()
    # sink = -1
    # while sink == -1:
    #     index = -1
    #     lowest = torch.inf
    #     SR[i] = True

    #     for it in range(num_remaining):
    #         j = remaining[it]
    #         r = min_val + cost[i, j] - u[i] - v[j]

    #         if r < shortest_path_costs[j]:
    #             path[j] = i
    #             shortest_path_costs[j] = r

    #         if (shortest_path_costs[j] < lowest or 
    #            (shortest_path_costs[j] == lowest and row4col[j] == -1)):
    #             lowest = shortest_path_costs[j]
    #             index = it

    #     min_val = lowest
    #     j = remaining[index]
    #     if min_val == torch.inf:
    #         return -1, min_val,path,row4col, shortest_path_costs,SR, SC

    #     if row4col[j] == -1:
    #         sink = j
    #     else:
    #         i = row4col[j]

    #     SC[j] = True
    #     mask = torch.ones(remaining.numel(), dtype=torch.bool,device=device)
    #     mask[index] = False
    #     remaining = remaining[mask]

    #     num_remaining -= 1
    device = cost.device
    nc = cost.shape[1]
    sink = torch.tensor(-1,device=device)
    min_val = torch.tensor(0,device=device)
    # remaining = torch.arange(nc - 1, -1, -1,device=device)
    modified_cost = cost.clone()

     # Clone cost matrix to modify it directly during execution
    modified_cost = cost.clone()
    update_mask=None
    while sink == -1:
        remaining = torch.arange(nc - 1, -1, -1,device=device)
        SR[i] = True

        # Calculate `r` for all `remaining` indices in a vectorized way
        r = min_val + modified_cost[i, remaining] - u[i] - v[remaining]

        # Update shortest path costs and path using the current row
        update_mask = r < shortest_path_costs[remaining]
        shortest_path_costs[remaining[update_mask]] = r[update_mask]
        path[remaining[update_mask]] = i

        # Find the lowest path cost among remaining columns and get the corresponding index
        min_remaining_costs = shortest_path_costs[remaining]
        min_val, index = torch.min(min_remaining_costs, dim=0)
        j = remaining[index]

        # Check for infeasible matrix
        if min_val == torch.inf:
            return -1, min_val, path, row4col, shortest_path_costs, SR, SC

        # Determine if we found a sink
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        # Update remaining columns and mark column as covered
        SC[j] = True
        remaining = remaining[remaining != j]  # Filter out selected index

    return sink, min_val, path, row4col, shortest_path_costs, SR, SC

    return sink, min_val, path, row4col, shortest_path_costs, SR, SC


def solve(input_cost):
    device = input_cost.device
    nr, nc = input_cost.shape
    cost = input_cost - torch.min(input_cost)

    u = torch.zeros(nr,device=device)
    v = torch.zeros(nc,device=device)
    shortest_path_costs = torch.full((nc,), torch.inf,device=device)
    path = torch.full((nc,), -1, dtype=torch.int,device=device)
    col4row = torch.full((nr,), -1, dtype=torch.int,device=device)
    row4col = torch.full((nc,), -1, dtype=torch.int,device=device)
    SR = torch.zeros(nr, dtype=torch.bool,device=device)
    SC = torch.zeros(nc, dtype=torch.bool,device=device)

    for cur_row in range(nr):
        min_val = 0
        sink, min_val,path,row4col,shortest_path_costs,SR, SC = augmenting_path(cost, u, v, path, row4col, shortest_path_costs, cur_row, SR, SC)

        if sink < 0:
            return -1

        u[cur_row] += min_val
        for i in range(nr):
            if SR[i] and i != cur_row:
                u[i] += min_val - shortest_path_costs[col4row[i]]
        for j in range(nc):
            if SC[j]:
                v[j] -= min_val - shortest_path_costs[j]

        j = sink
        while True:
            i = path[j]
            print('col4row[i]',col4row[i])
            row4col[j] = i
            temp_j=j.clone()
            j  = col4row[i].clone()
            col4row[i] = temp_j
            print('i',i)
            print('path',path)
            print('j',j)
            print('col4row[i] dopo',col4row[i])
            input()
            if (i == cur_row).item():
                break

    return col4row

def solve_rectangular_linear_sum_assignment(input_cost):
    col4row = solve(input_cost)
    not_finished=col4row is None 
    return None if not_finished else col4row



def calculate_assignment(cost_matrix):
    # Ensure the input is a NumPy array of type double and 2D
    cost_matrix = cost_matrix
    device=cost_matrix.device
    
    if cost_matrix.ndim != 2:
        raise TypeError("Invalid cost matrix object, must be a 2D array")

    # Get dimensions of the matrix
    num_rows, num_cols = cost_matrix.shape
    # Check for NaN or -inf entries
    if torch.isnan(cost_matrix).any() or torch.isneginf(cost_matrix).any():
        raise ValueError("Matrix contains invalid numeric entries")

    # Create an array `a` with indices of rows (0, 1, ..., num_rows-1)
    a = torch.arange(num_rows, dtype=torch.int,device=device)

    # Call the solve function to get the column assignment for each row
    try:
        if cost_matrix.shape[0]>cost_matrix.shape[1]:
            b = solve_rectangular_linear_sum_assignment(cost_matrix.T)

            indices = torch.argsort(b)
            
            return torch.as_tensor(b, dtype=torch.int,device=device)[indices],a[indices]
        else:
            b = solve_rectangular_linear_sum_assignment(cost_matrix)
            return a, torch.as_tensor(b, dtype=torch.int,device=device)
    except Exception:
        raise ValueError("Cost matrix is infeasible")

    # Package results in a tuple similar to Py_BuildValue("OO", a, b)
    


if __name__ == '__main__':
    import numpy as np
    # Example usage
    input_cost = np.array([[81, 30, 30, 51, 90, 63, 38, 21, 86, 39, 79, 12, 20, 23, 88, 89,
            81,  2, 26, 97],
        [70, 40, 97, 44, 38, 17, 51, 81, 94, 11, 75, 31,  5, 78, 88, 62,
            71,  7, 18, 37],
        [35, 16, 36, 57, 44, 61, 81, 61, 59, 60,  6, 62, 47, 18, 36, 54,
            52, 44,  7,  2],
        [87, 16, 96,  3, 17, 66, 41, 96, 34, 92, 50, 86, 35, 94, 36, 38,
            14, 42, 95,  2],
        [24, 27, 51,  6, 23, 24, 62, 64, 71, 10, 41, 49, 11, 47,  1, 94,
            0, 76, 75, 32],
        [38, 94, 27, 11, 59, 32, 49, 88, 84, 42, 53, 42, 86,  3, 32, 54,
            40, 26, 23, 17],
        [24, 13, 15, 76, 72, 52,  2, 46,  6, 62, 58, 63, 27, 56,  4, 59,
            80, 50, 69, 99],
        [29, 31, 85, 26, 43,  3, 65, 86, 95, 28, 39, 12, 61, 38, 49, 88,
            30, 38, 91,  7],
        [37,  2, 64,  7, 42, 70, 31, 46, 22,  2, 62, 73, 89, 96, 75, 68,
            95, 53, 29, 64],
        [41, 34, 47, 72, 50, 79, 29, 56, 74, 11, 68, 28, 53, 53, 25, 18,
            56, 56, 14, 17],
        [88, 17, 14, 51, 79, 15, 77, 87, 58, 37, 12, 10, 95, 53, 47, 76,
            70, 80, 94, 69],
        [44, 33, 74,  7, 79, 11, 98,  1, 98, 66, 59, 38, 79, 36, 98, 13,
            9, 82,  1, 46],
        [73, 18, 48, 19, 61, 73, 21, 95,  8, 55, 72,  4, 90, 66, 58, 79,
            43, 47,  5, 13],
        [50, 13, 94, 63, 75, 28,  5, 43, 54, 44, 14, 96, 68, 63, 39, 80,
            75, 75, 71, 29],
        [31, 76,  3, 76, 30, 84, 64, 32, 61, 25, 83, 65, 57, 99, 44, 59,
            11,  9, 97, 58],
        [55, 28, 38, 10, 89, 21, 73, 67, 15, 25, 47, 64, 93,  7,  0, 88,
            42, 37, 21, 31],
        [75,  3, 48, 14, 11, 42, 18, 60, 93, 81,  7, 91, 70,  3,  2,  8,
            8, 68, 82, 47],
        [44, 87, 94, 15, 23, 82, 99, 62, 65, 38, 51, 14, 16, 75, 94, 48,
            75, 78, 97, 68],
        [11, 47, 77, 17, 59,  9, 88, 78, 16, 95, 10, 91, 62, 18, 74, 85,
            79, 33, 82, 20],
        [41,  5, 24, 37, 30,  8, 95, 73,  4, 14, 10, 71, 39, 65, 86, 68,
            81, 83, 57, 73]])

    result = calculate_assignment(torch.from_numpy(input_cost).cuda())
    print("Column assignment for each row:", result)


    from scipy.optimize import linear_sum_assignment

    result2=linear_sum_assignment(input_cost)

    print("Column assignment for each row:", result2)


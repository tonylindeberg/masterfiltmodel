import argparse
import copy
import numpy as np
import torch
from data_tools.normalization import normalize
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Process model parameters.')
parser.add_argument('--model_type', type=str, required=False, help='Model type', default='convnextv2')
parser.add_argument('--trained_model_path', type=str, required=False, help='Path to the model file', default='/home/gpude06/pretrained/convnextv2_tiny_1k_224_ema.pt')
parser.add_argument('--model_path', type=str, required=False, help='Path to the model file', default='/home/gpude06/pretrained/convnextv2_tiny_1k_224_fcmae.pt')
parser.add_argument('--modified_model_path', type=str, required=False, help='Path to save the modified model', default='/home/gpude06/pretrained/convnextv2_tiny_1k_224_fcmae_methodB.pt')
parser.add_argument('--method', type=str, required=False, help='Mathematical method for modelling the Master Key Filters', default='B')

args = parser.parse_args()

model_type = args.model_type
trained_model_path = args.trained_model_path
model_path = args.model_path
modified_model_path = args.modified_model_path

model_trained = torch.load(trained_model_path)

model_original = torch.load(model_path)
model_modified = copy.deepcopy(model_original)

def find_min_norm_indices(mat_y, mat_x):
    """
    Finding the best estimates for the filters Y as a linear combination of the filters X.
    Y ~= aX + b = (Y.X)X + mean(Y)
    :param mat_y:
    :param mat_x:
    :return:
    """
    if torch.any(torch.abs(mat_x.mean(dim=1)) > 0.01) or torch.any(torch.abs(torch.linalg.norm(mat_x, dim=1) - 1) > 0.01):
        raise ValueError("The mean of X must be 0 and the norm of X must be 1.")

    # Compute the dot product between all pairs of rows from Y and X
    dot_products = torch.matmul(mat_y, mat_x.T)  # Shape (n, m)

    # Multiply each row in X by each dot product (scale X rows)
    # Resulting shape will be (n, m, k) after broadcasting
    mat_aX = dot_products.unsqueeze(2) * mat_x.unsqueeze(0)

    # Subtract scaled B from each row in A and add the mean of A
    mean_y = mat_y.mean(dim=1, keepdim=True).unsqueeze(2)
    y_estimate = mat_aX + mean_y
    distances = torch.linalg.norm(mat_y.unsqueeze(1) - y_estimate, dim=2)

    # Find the minimum values and their indices along axis 1
    min_values, argmin_indices = torch.min(distances, dim=1)

    x_tensors = []

    for i in range(mat_y.shape[0]):
        idx = argmin_indices[i]  # Index of the minimum distance
        x_tensors.append(mat_x[idx])  # Append the corresponding X tensor

        # Convert the list of tensors to a single tensor
    x_tensors = torch.stack(x_tensors)

    return argmin_indices, y_estimate[torch.arange(mat_y.shape[0]), argmin_indices], x_tensors


import torch
import torch.nn.functional as F
import numpy as np
from math import exp
from scipy.special import iv  # Modified Bessel function of the first kind

def discrete_gaussian_kernel_1d(sigma, size=7):
    """
    Implement the 1D discrete analogue of the Gaussian kernel using modified Bessel functions.
    T1D(n; σ) = e^(-σ²) * I_n(σ²)
    where I_n is the modified Bessel function of integer order.
    """
    # Ensure size is odd
    if size % 2 == 0:
        size += 1
    
    # Create a range of integer values centered at 0
    n = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
    
    # Compute the variance
    var = sigma**2
    
    # Initialize the kernel
    kernel = torch.zeros_like(n)
    
    # Calculate using the formula T1D(n; σ) = e^(-σ²) * I_n(σ²)
    for i, ni in enumerate(n):
        # Use scipy's modified Bessel function
        bessel_value = iv(int(abs(ni.item())), var)
        kernel[i] = exp(-var) * bessel_value
    
    # Normalize the kernel
    return kernel / kernel.sum()

def discrete_gaussian_kernel_2d(sigma_x, sigma_y, size=7):
    """
    Create a 2D discrete Gaussian kernel as a separable product of 1D kernels.
    T(m, n; σx, σy) = T1D(m; σx) * T1D(n; σy)
    """
    # Create 1D kernels
    kernel_x = discrete_gaussian_kernel_1d(sigma_x, size)
    kernel_y = discrete_gaussian_kernel_1d(sigma_y, size)
    
    # Create 2D kernel through outer product
    kernel_2d = torch.outer(kernel_y, kernel_x)
    
    return kernel_2d

def non_centered_difference_op(kernel_2d, direction, sign):
    """
    Apply a non-centered first-order difference operator to a 2D kernel.
    direction: 'x' or 'y'
    sign: '+' or '-'
    """
    h, w = kernel_2d.shape
    result = torch.zeros_like(kernel_2d)
    
    if direction == 'x':
        if sign == '+':
            # δx+ (forward difference)
            result[:, :-1] = kernel_2d[:, 1:] - kernel_2d[:, :-1]
        else:  # sign == '-'
            # δx- (backward difference)
            result[:, 1:] = kernel_2d[:, 1:] - kernel_2d[:, :-1]
    else:  # direction == 'y'
        if sign == '+':
            # δy+ (forward difference)
            result[:-1, :] = kernel_2d[1:, :] - kernel_2d[:-1, :]
        else:  # sign == '-'
            # δy- (backward difference)
            result[1:, :] = kernel_2d[1:, :] - kernel_2d[:-1, :]
    
    return result

def centered_difference_op_full(kernel_2d, direction):
    """
    Centered difference in the interior, and one-sided differences at boundaries
    so NO row/column stays identically zero.
    """
    result = torch.zeros_like(kernel_2d)

    if direction == 'x':
        # interior centered
        result[:, 1:-1] = (kernel_2d[:, 2:] - kernel_2d[:, :-2]) / 2
        # left boundary: forward difference
        result[:, 0] = kernel_2d[:, 1] - kernel_2d[:, 0]
        # right boundary: backward difference
        result[:, -1] = kernel_2d[:, -1] - kernel_2d[:, -2]
    else:  # 'y'
        # interior centered
        result[1:-1, :] = (kernel_2d[2:, :] - kernel_2d[:-2, :]) / 2
        # top boundary: forward difference
        result[0, :] = kernel_2d[1, :] - kernel_2d[0, :]
        # bottom boundary: backward difference
        result[-1, :] = kernel_2d[-1, :] - kernel_2d[-2, :]

    return result


def centered_difference_op(kernel_2d, direction):
    """
    Apply a centered first-order difference operator to a 2D kernel.
    direction: 'x' or 'y'
    """
    h, w = kernel_2d.shape
    result = torch.zeros_like(kernel_2d)
    
    if direction == 'x':
        # δx (centered difference)
        result[:, 1:-1] = (kernel_2d[:, 2:] - kernel_2d[:, :-2]) / 2
    else:  # direction == 'y'
        # δy (centered difference)
        result[1:-1, :] = (kernel_2d[2:, :] - kernel_2d[:-2, :]) / 2
    
    return result

def laplacian_op(kernel_2d):
    """
    Apply the five-point discrete approximation of the Laplacian operator.
    ∇²₅ = δxx + δyy
    """
    h, w = kernel_2d.shape
    result = torch.zeros_like(kernel_2d)
    
    # Five-point stencil for Laplacian
    for i in range(1, h-1):
        for j in range(1, w-1):
            result[i, j] = (kernel_2d[i-1, j] + kernel_2d[i+1, j] + 
                           kernel_2d[i, j-1] + kernel_2d[i, j+1] - 
                           4 * kernel_2d[i, j])
    
    return result

def generate_filters(method='A', gamma7=0.5, size=7):
    """
    Generate the 8 master key filters for a specific method (A, B, C1, C2, D1, or D2).
    """
    # Define scale parameters for each method
    if method == 'A':
        scales = [
            (0.580, 0.558),  # Filter 1
            (0.559, 0.580),  # Filter 2
            (0.617, 0.601),  # Filter 3
            (0.642, 0.424),  # Filter 4
            (1.075, 0.800),  # Filter 5
            (0.771, 1.003),  # Filter 6
            (0.675, 0.675),  # Filter 7 
            (0.552, 0.545)   # Filter 8
        ]
    elif method == 'B':
        scales = [
            (0.644, 0.583),  # Filter 1
            (0.586, 0.644),  # Filter 2
            (0.690, 0.674),  # Filter 3
            (0.756, 0.460),  # Filter 4
            (1.107, 0.945),  # Filter 5
            (0.900, 0.889),  # Filter 6
            (0.675, 0.675),  # Filter 7 
            (0.609, 0.601)   # Filter 8
        ]
    elif method == 'C1':
        scales = [
            (0.360, 0.510),  # Filter 1
            (0.555, 0.453),  # Filter 2
            (0.701, 0.655),  # Filter 3
            (0.563, 0.384),  # Filter 4
            (1.309, 0.875),  # Filter 5
            (0.973, 1.171),  # Filter 6
            (0.654, 0.654),  # Filter 7 
            (0.637, 0.587)   # Filter 8
        ]
    elif method == 'C2':
        scales = [
            (0.458, 0.458),  # Filter 1
            (0.448, 0.448),  # Filter 2
            (0.671, 0.671),  # Filter 3
            (0.420, 0.420),  # Filter 4
            (1.387, 1.387),  # Filter 5
            (1.090, 1.090),  # Filter 6
            (0.654, 0.654),  # Filter 7
            (0.611, 0.611)   # Filter 8
        ]
    elif method == 'D1':
        scales = [
            (0.491, 0.722),  # Filter 1
            (0.581, 0.519),  # Filter 2
            (0.483, 0.503),  # Filter 3
            (0.500, 0.000),  # Filter 4 
            (1.300, 1.004),  # Filter 5
            (0.984, 1.074),  # Filter 6
            (0.675, 0.675),  # Filter 7 
            (0.615, 0.608)   # Filter 8
        ]
    elif method == 'D2':
        scales = [
            (0.644, 0.644),  # Filter 1
            (0.558, 0.558),  # Filter 2
            (0.495, 0.495),  # Filter 3
            (0.380, 0.380),  # Filter 4
            (1.193, 1.193),  # Filter 5
            (1.038, 1.038),  # Filter 6
            (0.675, 0.675),  # Filter 7
            (0.612, 0.612)   # Filter 8
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    filters = []
    
    # Generate each filter
    for i, (sigma_x, sigma_y) in enumerate(scales):
        # Handle the special case of D1 Filter 4 where sigma_y = 0
        if method == 'D1' and i == 3 and sigma_y == 0:
            sigma_y = 1e-6  # Use a very small value instead of 0
            
        # Generate the base Gaussian kernel
        gaussian = discrete_gaussian_kernel_2d(sigma_x, sigma_y, size)
        
        # Apply the appropriate operator for each filter
        if i == 0:  # Filter 1: (δy+T)
            filter_kernel = non_centered_difference_op(gaussian, 'y', '+')
        elif i == 1:  # Filter 2: (δx-T)
            filter_kernel = non_centered_difference_op(gaussian, 'x', '-')
        elif i == 2:  # Filter 3: (δy-T)
            filter_kernel = non_centered_difference_op(gaussian, 'y', '-')
        elif i == 3:  # Filter 4: (δx+T)
            filter_kernel = non_centered_difference_op(gaussian, 'x', '+')
        elif i == 4:  # Filter 5: (δxT)
            filter_kernel = centered_difference_op_full(gaussian, 'x')
        elif i == 5:  # Filter 6: (δyT)
            filter_kernel = centered_difference_op_full(gaussian, 'y')
        elif i == 6:  # Filter 7: 1 - γ7(∇²₅T)
            laplacian = laplacian_op(gaussian)
            filter_kernel = gaussian - gamma7 * laplacian
        else:  # Filter 8: T (pure smoothing)
            filter_kernel = gaussian
        
        # Normalize each filter to have unit Frobenius norm
        if torch.norm(filter_kernel) > 0:
            filter_kernel = filter_kernel / torch.norm(filter_kernel)
        
        filters.append(filter_kernel)
    
    return filters


# for method in ['A', 'B', 'C1', 'C2', 'D1', 'D2']:
filters = generate_filters(method=args.method)

#candidate_filters = torch.tensor(filters, dtype=torch.float32).view(8, 49)
candidate_filters = torch.stack(filters).float().view(8, 49)


total_counts = torch.zeros(len(candidate_filters), dtype=torch.int64)

for param_name in model_trained['model'].keys():
    if 'dwconv.weight' in param_name:
        filters = model_trained['model'][param_name].clone()

        filters_np = filters.squeeze(1).detach().cpu().numpy()
        filters_norm_np = normalize(filters_np, "center_norm")
        filters_norm = torch.from_numpy(filters_norm_np).to(filters.device)
        flat_filters_norm = filters_norm.flatten(start_dim=1)

        # filters_norm = normalize(np.array(filters.squeeze(1)), "center_norm")
        # flat_filters_norm = torch.tensor(filters_norm).flatten(start_dim=1)
        flat_filters_original = filters.clone().flatten(start_dim=1)

        candidates = candidate_filters - torch.mean(candidate_filters, dim=1, keepdim=True)
        candidates = candidates / torch.norm(candidates, dim=1, keepdim=True)

        min_distance_indices, best_estimates, best_estimates_original = find_min_norm_indices(flat_filters_original, candidates)
        print(best_estimates.shape)
        counts = torch.bincount(min_distance_indices, minlength=len(candidate_filters))
        total_counts += counts

        # Replace the filters with the modified filters
        model_modified['model'][param_name] = best_estimates.reshape(filters.shape)

        #model_modified['model'][param_name] = best_estimates_original.reshape(filters.shape)

print(total_counts)
torch.save(model_modified, modified_model_path)

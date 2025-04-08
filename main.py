import torch
from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

import math
import os


def partition_grid(grid: torch.Tensor, D, E, F):
    """
    Partition 3d tensor into (A//D)*(B//E)*(C//F) subblocks
    """

    A, B, C = grid.shape
    dx, dy, dz = A//D, B//E, C//F

    # reshape
    partitioned = grid.view(
        D, dx,
        E, dy,
        F, dz
    )
    # group block indices first
    partitioned = partitioned.permute(0, 2, 4, 1, 3, 5)

    # returns tensor witih transforms
    return partitioned.contiguous()


def chebyshev_basis(grid: torch.Tensor, order=3):
    D, E, F, dx, dy, dz = grid.shape

    # normalized coords per dim
    x = torch.linspace(-1, 1, dx)
    y = torch.linspace(-1, 1, dy)
    z = torch.linspace(-1, 1, dz)

    Tx = make_1d_basis(x, order) # [order+1, dx]
    Ty = make_1d_basis(y, order)
    Tz = make_1d_basis(z, order)

    basis = (
        Tx[:, None, None, :, None, None] * # T_i(x)
        Ty[None, :, None, None, :, None] *
        Tz[None, None, :, None, None, :]
    )

    # flatten + block dimensions
    basis = basis.reshape((order + 1) ** 3, dx, dy, dz)[None, None, None] \
                .expand(D, E, F, -1, dx, dy, dz)

    return basis


def make_1d_basis(coords, n_max):
    """ Generates 1D chebyshev polynomial of first kind """
    return torch.stack([
        torch.special.chebyshev_polynomial_t(coords, n)
        for n in range(n_max + 1)
    ], dim=0)


def compute_chebyshev_coefficients(partitioned_data: torch.Tensor, chebyshev_basis: torch.Tensor):
    # Reshape preparations
    D, E, F, dx, dy, dz = partitioned_data.shape
    G = chebyshev_basis.shape[3]
    batch_size = D*E*F
    N = dx*dy*dz

    # Reshape basis and data
    A = chebyshev_basis.reshape(batch_size, G, N).permute(0, 2, 1)  # [batch, N, G]
    b = partitioned_data.reshape(batch_size, N, 1)

    # QR decomposition
    Q, R = torch.linalg.qr(A, mode='reduced')  # Q: [batch, N, k], R: [batch, k, G] where k=min(N,G)

    # Compute Q^T·b
    Qtb = torch.bmm(Q.transpose(1, 2), b)  # [batch, k, 1]

    # Solve R·x = Q^T·b using triangular solve if R is square
    # For non-square case, use pseudo-inverse
    if N >= G:  # R is [batch, G, G]
        solution = torch.linalg.solve_triangular(R, Qtb, upper=True)
    else:
        # For overdetermined system (N < G), use pseudo-inverse
        solution = torch.linalg.lstsq(R, Qtb).solution

    """
    # Flatten spatial dimensions for each block
    A = chebyshev_basis.permute(0,1,2,3,4,5,6).reshape(D*E*F, G, dx*dy*dz)
    b = partitioned_data.reshape(D*E*F, dx*dy*dz, 1)

    # QR decomposition with numerical stability
    Q, R = torch.linalg.qr(A.permute(0,2,1), mode='reduced')  # [B, N, G]

    # Solve R^T c = Q^T b (equivalent to least squares)
    Qtb = torch.bmm(Q.transpose(1,2), b)  # [B, G, 1]
    solution = torch.linalg.solve_triangular(R, Qtb, upper=True)  # [B, G, 1]
    """

    return solution.reshape(D, E, F, G).contiguous()


def reassemble_grid(coefficients: torch.Tensor,
                   chebyshev_basis: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs original grid from Chebyshev coefficients and basis
    Implements the reconstruction formula:
    reconstructed_block = Σ (coefficients_g * basis_g)
                          for g=0 to G-1
    """
    # assert coefficients.dim() == 4, "Coefficients must be 4D [D,E,F,G]"
    # assert chebyshev_basis.dim() == 7, "Basis must be 7D [D,E,F,G,dx,dy,dz]"

    D, E, F, G = coefficients.shape
    _, _, _, _, dx, dy, dz = chebyshev_basis.shape

    # 1. Compute block reconstructions using tensor contraction
    # [D,E,F,G] @ [D,E,F,G,dx,dy,dz] → [D,E,F,dx,dy,dz]
    reconstructed_blocks = torch.einsum('defg,defgxyz->defxyz',
                                      coefficients,
                                      chebyshev_basis)

    # 2. Recombine blocks into full grid
    # Permute to [D, dx, E, dy, F, dz] then flatten
    reconstructed = reconstructed_blocks.permute(0, 3, 1, 4, 2, 5) \
                                        .contiguous() \
                                        .view(D*dx, E*dy, F*dz)

    return reconstructed


def lagrange_basis_1d(x, nodes):
    """Dimension-robust 1D basis construction"""
    n = nodes.size(0)
    basis = torch.ones(x.size(0), n, device=x.device)
    for i in range(n):
        for j in range(n):
            if i != j:
                basis[:, i] *= (x - nodes[j]) / (nodes[i] - nodes[j] + 1e-12)  # Prevent div0
    return basis


def lagrange_compress_3d(grid, D, E, F, p):
    """
    Universal Lagrange compression for any divisible dimensions
    Args:
        grid: Input tensor [A,B,C]
        D,E,F: Target partition counts (must exactly divide A,B,C)
        p: Polynomial degree
    Returns:
        Compressed tensor [D,E,F,(p+1)^3]
    """
    A, B, C = grid.shape
    dx, dy, dz = A//D, B//E, C//F
    G = (p+1)**3

    # Validate exact division
    assert A == D*dx, f"A({A}) must be divisible by D({D})"
    assert B == E*dy, f"B({B}) must be divisible by E({E})"
    assert C == F*dz, f"C({C}) must be divisible by F({F})"

    # Generate normalized coordinates [-1,1]
    def linspace(d):
        return torch.linspace(-1, 1, d, device=grid.device)
    x, y, z = linspace(dx), linspace(dy), linspace(dz)

    # Chebyshev-like nodes for stability
    nodes = torch.cos(torch.pi*(0.5 + torch.arange(p+1))/(p+1))

    # Uniform nodes
    # nodes = torch.linspace(-1, 1, p+1)

    # Build 3D basis matrix [block_points × basis_functions]
    basis_x = lagrange_basis_1d(x, nodes)  # [dx × (p+1)]
    basis_y = lagrange_basis_1d(y, nodes)  # [dy × (p+1)]
    basis_z = lagrange_basis_1d(z, nodes)  # [dz × (p+1)]

    # Tensor product basis [dx×dy×dz × (p+1)^3]
    full_basis = torch.einsum('xi,yj,zk->xyzijk', basis_x, basis_y, basis_z)
    full_basis = full_basis.reshape(-1, G)  # [block_points × G]

    # Universal compression formula
    compressed = (
        grid.view(D, dx, E, dy, F, dz)    # Split into blocks
        .permute(0, 2, 4, 1, 3, 5)        # [D,E,F,dx,dy,dz]
        .reshape(D*E*F, -1)               # [num_blocks × block_points]
        @ torch.linalg.pinv(full_basis).T  # [block_points × G] → [block_points × G]
    )

    return compressed.reshape(D, E, F, G)


def lagrange_decompress_3d(compressed, original_shape, p):
    """
    Decompress D×E×F×G tensor back to original A×B×C grid

    Args:
        compressed: Compressed tensor [D,E,F,G]
        original_shape: Tuple (A,B,C) of target grid dimensions
        p: Polynomial degree used during compression
    Returns:
        Decompressed tensor [A,B,C]
    """
    D, E, F, G = compressed.shape
    A, B, C = original_shape
    dx, dy, dz = A//D, B//E, C//F

    # Generate normalized coordinates [-1,1] for each block
    x = torch.linspace(-1, 1, dx, device=compressed.device)
    y = torch.linspace(-1, 1, dy, device=compressed.device)
    z = torch.linspace(-1, 1, dz, device=compressed.device)

    # Recreate Lagrange nodes (must match compression nodes)
    nodes = torch.cos(torch.pi*(0.5 + torch.arange(p+1))/(p+1))

    # Reconstruct basis matrices
    basis_x = lagrange_basis_1d(x, nodes)
    basis_y = lagrange_basis_1d(y, nodes)
    basis_z = lagrange_basis_1d(z, nodes)

    # Create 3D basis tensor [dx,dy,dz,G]
    full_basis = torch.einsum('xi,yj,zk->xyzijk', basis_x, basis_y, basis_z) \
                     .reshape(dx*dy*dz, -1)

    # Reconstruct blocks and reshape to original grid
    blocks = compressed.reshape(D*E*F, G) @ full_basis.T
    return blocks.reshape(D,E,F,dx,dy,dz) \
                .permute(0,3,1,4,2,5) \
                .contiguous() \
                .view(A,B,C)


def gaussian_basis_3d(partitioned_grid, sigma=0.5, n_centers=3):
    """
    Generates 3D Gaussian basis functions for partitioned grid

    Args:
        partitioned_grid: Tensor from torch_partition()
                          Shape [D, E, F, dx, dy, dz]
        sigma: Width parameter for Gaussian functions (default=0.5)
        n_centers: Number of centers per dimension (default=3)
                   Total basis functions = n_centers^3

    Returns:
        basis: Tensor of shape [D, E, F, G, dx, dy, dz]
               Contains Gaussian basis evaluations for each subblock
    """
    D, E, F, dx, dy, dz = partitioned_grid.shape
    device = partitioned_grid.device
    G = n_centers**3  # Total number of basis functions

    # Generate normalized coordinates [-1, 1] for each dimension
    x = torch.linspace(-1, 1, dx, device=device)
    y = torch.linspace(-1, 1, dy, device=device)
    z = torch.linspace(-1, 1, dz, device=device)

    # Create centers for Gaussian functions (evenly spaced in [-1, 1])
    centers = torch.linspace(-1, 1, n_centers, device=device)

    # Create meshgrid of coordinates
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Initialize basis tensor
    basis = torch.zeros(D, E, F, G, dx, dy, dz, device=device)

    # Compute Gaussian basis functions
    idx = 0
    for cx in centers:
        for cy in centers:
            for cz in centers:
                # Compute squared distance from center
                dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2

                # Evaluate Gaussian function: exp(-dist²/2σ²)
                gaussian = torch.exp(-dist_sq/(2*sigma**2))

                # Add to basis tensor (broadcast across all blocks)
                basis[:, :, :, idx] = gaussian
                idx += 1

    return basis

def gaussian_compress_3d(grid, D, E, F, n_centers=3, sigma=0.5):
    """
    Compresses 3D grid using Gaussian basis functions

    Args:
        grid: Input tensor [A,B,C]
        D,E,F: Target partition counts (must divide A,B,C)
        n_centers: Number of centers per dimension
        sigma: Width parameter for Gaussian functions

    Returns:
        Compressed tensor [D,E,F,G] where G = n_centers^3
    """
    A, B, C = grid.shape
    dx, dy, dz = A//D, B//E, C//F
    G = n_centers**3

    # Validate exact division
    assert A == D*dx, f"A({A}) must be divisible by D({D})"
    assert B == E*dy, f"B({B}) must be divisible by E({E})"
    assert C == F*dz, f"C({C}) must be divisible by F({F})"

    # Partition the grid
    partitioned = grid.view(D, dx, E, dy, F, dz).permute(0, 2, 4, 1, 3, 5)

    # Generate Gaussian basis
    basis = gaussian_basis_3d(partitioned, sigma, n_centers)

    # Reshape for least squares solve
    batch_size = D*E*F
    n_points = dx*dy*dz

    # Reshape basis to [batch, points, G]
    A_mat = basis.permute(0, 1, 2, 4, 5, 6, 3).reshape(batch_size, n_points, G)

    # Reshape data to [batch, points, 1]
    b_vec = partitioned.reshape(batch_size, n_points, 1)

    # Solve least squares problem for each block
    coefficients = torch.linalg.lstsq(A_mat, b_vec).solution.squeeze(-1)

    return coefficients.reshape(D, E, F, G)


def gaussian_decompress_3d(coefficients, original_shape, n_centers=3, sigma=0.5):
    """
    Decompresses coefficients back to original grid

    Args:
        coefficients: Compressed tensor [D,E,F,G]
        original_shape: Tuple (A,B,C) of target grid dimensions
        n_centers: Number of centers per dimension
        sigma: Width parameter for Gaussian functions

    Returns:
        Decompressed tensor [A,B,C]
    """
    D, E, F, G = coefficients.shape
    A, B, C = original_shape
    dx, dy, dz = A//D, B//E, C//F

    # Create empty partitioned grid to generate basis
    partitioned = torch.zeros(D, E, F, dx, dy, dz, device=coefficients.device)

    # Generate Gaussian basis
    basis = gaussian_basis_3d(partitioned, sigma, n_centers)

    # Reconstruct blocks using coefficients
    reconstructed = torch.einsum('defg,defgxyz->defxyz', coefficients, basis)

    # Reshape to original grid dimensions
    return reconstructed.permute(0, 3, 1, 4, 2, 5).reshape(A, B, C)


def mae(actual: torch.Tensor, predicted: torch.Tensor):
    return torch.mean(torch.abs(actual - predicted))


def mean_percentage_diff(actual: torch.Tensor, predicted: torch.Tensor):
    return torch.sum(torch.abs(actual - predicted) / torch.sum(actual)) * 1000


def parse_chgcar(chgcar_fn, no_data_fn):

    charge, mag = [], []
    dims = None
    is_charge, is_mag = False, False

    with open(chgcar_fn, "r") as fi:

        lines = fi.read().splitlines()
        fo = open(no_data_fn, "w")

        i = 0
        while i < len(lines):
            if not is_charge and not is_mag:
                fo.write(lines[i] + "\n")

            if not charge and not lines[i].strip():
                dims = lines[i + 1]
                is_charge = True
                fo.write(lines[i + 1] + "\n")
                i += 1
            elif is_charge and "augmentation" not in lines[i]:
                line_nums = lines[i].strip().split(" ")
                for num in line_nums:
                    charge.append(float(num))
            elif is_charge and "augmentation" in lines[i]:
                fo.write(lines[i] + "\n")
                is_charge = False
            elif not mag and dims == lines[i]:
                is_mag = True
            elif is_mag and "augmentation" not in lines[i]:
                line_nums = lines[i].strip().split(" ")
                for num in line_nums:
                    mag.append(float(num))
            elif is_mag and "augmentation" in lines[i]:
                fo.write(lines[i] + "\n")
                is_mag = False

            i += 1

        fo.close()

    dims = [int(dim) for dim in dims.strip().split()]

    return dims, charge, mag


def pad_grid(grid: torch.Tensor, block_dims: list[int, 3]) -> tuple[torch.Tensor, tuple]:
    """
    Args:
        grid: 3D tensor (D, H, W)
        block_dims: Tuple of (depth_block, height_block, width_block)
    """
    original_shape = grid.shape
    padding_meta = []
    pad_values = []

    # Process dimensions in reverse order (width, height, depth)
    for i in reversed(range(3)):
        dim_size = original_shape[i]
        block = block_dims[i]
        remainder = dim_size % block

        if remainder == 0:
            pad_total = 0
        else:
            pad_total = block - remainder

        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        padding_meta.insert(0, pad_total)
        pad_values.extend([pad_before, pad_after])

    padded = torch.nn.functional.pad(grid, pad_values)
    return padded


def unpad_grid(padded: torch.Tensor,
                    original_shape: tuple) -> torch.Tensor:
    """
    Reverses the smart padding
    """
    slices = []
    for dim_size, padded_dim_size in zip(original_shape, padded.shape):
        if padded_dim_size == dim_size:
            slices.append(slice(None))
        else:
            pad_total = padded_dim_size - dim_size
            pad_start = pad_total // 2
            pad_end = pad_total - pad_start
            slices.append(slice(pad_start, dim_size + pad_start))

    return padded[slices[0], slices[1], slices[2]]


def main():
    PATH = "chgcars/"
    with os.scandir(PATH) as it:
        for chgcar in it:
            # if "1219259" in chgcar.name:
                dims, charge, mag = parse_chgcar(chgcar, chgcar.name.split('.')[0] + "_no_data.txt")
                charge, mag = torch.tensor(charge).reshape(dims), torch.tensor(mag).reshape(dims)

                # TODO: Add padding for datasets not divisible by 4 (for example)

                FACTOR, ORDER = 4, 3
                block_dims = [math.ceil(dim / FACTOR) for dim in dims]
                charge_padded = pad_grid(charge, block_dims)
                mag_padded = pad_grid(mag, block_dims)
                padded_dims = charge_padded.shape

                # """ Chebyshev

                partitioned_data_charge = partition_grid(charge_padded, *block_dims)
                basis_charge = chebyshev_basis(partitioned_data_charge, order=ORDER)
                coeffs_charge = compute_chebyshev_coefficients(partitioned_data_charge, basis_charge)
                reassembled_data_charge = reassemble_grid(coeffs_charge, basis_charge)
                reassembled_data_charge = unpad_grid(reassembled_data_charge, dims)

                partitioned_data_mag = partition_grid(mag_padded, *block_dims)
                basis_mag = chebyshev_basis(partitioned_data_mag, order=ORDER)
                coeffs_mag = compute_chebyshev_coefficients(partitioned_data_mag, basis_mag)
                reassembled_data_mag = reassemble_grid(coeffs_mag, basis_mag)
                reassembled_data_mag = unpad_grid(reassembled_data_mag, dims)

                print(f"{chgcar}, dims: {dims}, coeffs dims: {coeffs_charge.shape}, charge mae: {mae(charge, reassembled_data_charge)}, charge mean_percetage_diff: {mean_percentage_diff(charge, reassembled_data_charge)}, mag mae: {mae(mag, reassembled_data_mag)}, mag mean_percentage_diff: {mean_percentage_diff(mag, reassembled_data_mag)}")
                # """

                """ Lagrange
                compressed_data_charge = lagrange_compress_3d(charge_padded, *block_dims, ORDER)
                decompressed_data_charge = lagrange_decompress_3d(compressed_data_charge, padded_dims, ORDER)
                decompressed_data_charge = unpad_grid(decompressed_data_charge, dims)

                compressed_data_mag = lagrange_compress_3d(mag_padded, *block_dims, ORDER)
                decompressed_data_mag = lagrange_decompress_3d(compressed_data_mag, padded_dims, ORDER)
                decompressed_data_mag = unpad_grid(decompressed_data_mag, dims)

                print(f"{chgcar}, dims: {dims}, compressed dims: {compressed_data_charge.shape}, charge mae: {mae(charge, decompressed_data_charge)}, charge mean_percetage_diff: {mean_percentage_diff(charge, decompressed_data_charge)}, mag mae: {mae(mag, decompressed_data_mag)}, mag mean_percetage_diff: {mean_percentage_diff(mag, decompressed_data_mag)}")
                """

                """ Gaussian
                compressed_data_charge = gaussian_compress_3d(charge_padded, *block_dims, ORDER)
                decompressed_data_charge = gaussian_decompress_3d(compressed_data_charge, padded_dims, ORDER)
                decompressed_data_charge = unpad_grid(decompressed_data_charge, dims)

                compressed_data_mag = gaussian_compress_3d(mag_padded, *block_dims, ORDER)
                decompressed_data_mag = gaussian_decompress_3d(compressed_data_mag, padded_dims, ORDER)
                decompressed_data_mag = unpad_grid(decompressed_data_mag, dims)

                print(f"{chgcar}, dims: {dims}, compressed dims: {compressed_data_charge.shape}, charge mae: {mae(charge, decompressed_data_charge)}, charge mean_percetage_diff: {mean_percentage_diff(charge, decompressed_data_charge)}, mag mae: {mae(mag, decompressed_data_mag)}, mag mean_percetage_diff: {mean_percentage_diff(mag, decompressed_data_mag)}")
                """


if __name__ == "__main__":
    main()

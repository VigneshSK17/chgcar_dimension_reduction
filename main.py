import click
import math
import os

import utils


@click.command()
@click.option("--factor", default=4, help="Divisor by which you wish to reduce dimensions of chgcar")
@click.option("--degree", default=3, help="The degree of the 4th dimension of data created during dimension reduction")
@click.option("--degree", default=3, help="The degree of the 4th dimension of data created during dimension reduction")
@click.option("--mode", type=click.Choice(['chebyshev', 'lagrange', 'gaussian']), required = True)
@click.argument("path")
def main(factor, degree, mode, path):
    """ Reduce dimensions of chgcars located in PATH by FACTOR with DEGREE """

    if os.path.isdir(path):
        with os.scandir(path) as it:
            for chgcar in it:
                # if "1219259" in chgcar.name:
                    dims, charge, mag = utils.parse_chgcar(chgcar, chgcar.name.split('.')[0] + "_no_data.txt")
                    charge, mag = utils.reshape_dims(charge, dims), utils.reshape_dims(mag, dims)

                    # TODO: Add padding for datasets not divisible by 4 (for example)

                    FACTOR, ORDER = 4, 3
                    block_dims = [math.ceil(dim / FACTOR) for dim in dims]
                    charge_padded = utils.pad_grid(charge, block_dims)
                    mag_padded = utils.pad_grid(mag, block_dims)
                    padded_dims = charge_padded.shape

                    # """ Chebyshev
                    if mode == 'chebyshev':
                        partitioned_data_charge = utils.partition_grid(charge_padded, *block_dims)
                        basis_charge = utils.chebyshev_basis(partitioned_data_charge, order=ORDER)
                        coeffs_charge = utils.compute_chebyshev_coefficients(partitioned_data_charge, basis_charge)
                        reassembled_data_charge = utils.reassemble_grid(coeffs_charge, basis_charge)
                        reassembled_data_charge = utils.unpad_grid(reassembled_data_charge, dims)

                        partitioned_data_mag = utils.partition_grid(mag_padded, *block_dims)
                        basis_mag = utils.chebyshev_basis(partitioned_data_mag, order=ORDER)
                        coeffs_mag = utils.compute_chebyshev_coefficients(partitioned_data_mag, basis_mag)
                        reassembled_data_mag = utils.reassemble_grid(coeffs_mag, basis_mag)
                        reassembled_data_mag = utils.unpad_grid(reassembled_data_mag, dims)

                        print(f"{chgcar}, dims: {dims}, coeffs dims: {coeffs_charge.shape}, charge mae: {utils.mae(charge, reassembled_data_charge)}, charge mean_percetage_diff: {utils.mean_percentage_diff(charge, reassembled_data_charge)}, mag mae: {utils.mae(mag, reassembled_data_mag)}, mag mean_percentage_diff: {utils.mean_percentage_diff(mag, reassembled_data_mag)}")


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

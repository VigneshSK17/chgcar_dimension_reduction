import click
import math
import os
import csv

from utils import *


@click.command()
@click.option("--factor", default=4, help="Divisor by which you wish to reduce dimensions of chgcar")
@click.option("--degree", default=3, help="The degree of the 4th dimension of data created during dimension reduction")
@click.option("--degree", default=3, help="The degree of the 4th dimension of data created during dimension reduction")
@click.option("--mode", type=click.Choice(['chebyshev', 'lagrange', 'gaussian']), required = True)
@click.option("--output", help="Path to save results in CSV format")
@click.argument("path")
def main(factor, degree, mode, output, path):
    """ Reduce dimensions of chgcars located in PATH by FACTOR with DEGREE """

    # Initialize CSV file if specified
    if output:
        csv_file = None
        csv_writer = None

        csv_file = open(output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write header row
        csv_writer.writerow([
            'filename', 'mode',
            'dim_x', 'dim_y', 'dim_z',
            'compressed_dim_x', 'compressed_dim_y', 'compressed_dim_z', 'compressed_dim_w',
            'charge_mae', 'charge_mean_percentage_diff',
            'mag_mae', 'mag_mean_percentage_diff'
        ])

    if os.path.isdir(path):
        with os.scandir(path) as it:
            for chgcar in it:
                if ".vasp" in chgcar.name:
                    dims, charge, mag = parse_chgcar(chgcar, chgcar.name.split('.')[0] + "_no_data.txt")
                    charge, mag = reshape_dims(charge, dims), reshape_dims(mag, dims)

                    block_dims = [math.ceil(dim / factor) for dim in dims]
                    charge_padded = pad_grid(charge, block_dims)
                    mag_padded = pad_grid(mag, block_dims)
                    padded_dims = charge_padded.shape

                    # """ Chebyshev
                    if mode == 'chebyshev':
                        partitioned_data_charge = partition_grid(charge_padded, *block_dims)
                        basis_charge = chebyshev_basis(partitioned_data_charge, order=degree)
                        coeffs_charge = compute_chebyshev_coefficients(partitioned_data_charge, basis_charge)
                        reassembled_data_charge = reassemble_grid(coeffs_charge, basis_charge)
                        reassembled_data_charge = unpad_grid(reassembled_data_charge, dims)

                        partitioned_data_mag = partition_grid(mag_padded, *block_dims)
                        basis_mag = chebyshev_basis(partitioned_data_mag, order=degree)
                        coeffs_mag = compute_chebyshev_coefficients(partitioned_data_mag, basis_mag)
                        reassembled_data_mag = reassemble_grid(coeffs_mag, basis_mag)
                        reassembled_data_mag = unpad_grid(reassembled_data_mag, dims)

                        charge_mae_val = mae(charge, reassembled_data_charge)
                        charge_mpd = mean_percentage_diff(charge, reassembled_data_charge)
                        mag_mae_val = mae(mag, reassembled_data_mag)
                        mag_mpd = mean_percentage_diff(mag, reassembled_data_mag)

                        print(f"{chgcar}, dims: {dims}, coeffs dims: {coeffs_charge.shape}, charge mae: {charge_mae_val}, charge mean_percetage_diff: {charge_mpd}, mag mae: {mag_mae_val}, mag mean_percentage_diff: {mag_mpd}")

                        if csv_writer:
                            csv_writer.writerow([
                                chgcar.name, 'chebyshev',
                                dims[0], dims[1], dims[2],
                                coeffs_charge.shape[0], coeffs_charge.shape[1], coeffs_charge.shape[2], coeffs_charge.shape[3],
                                f"{charge_mae_val.item():.4e}", f"{charge_mpd.item():.4e}",
                                f"{mag_mae_val.item():.4e}", f"{mag_mpd.item():.4e}"
                            ])

                    if mode == "lagrange":
                        compressed_data_charge = lagrange_compress_3d(charge_padded, *block_dims, degree)
                        decompressed_data_charge = lagrange_decompress_3d(compressed_data_charge, padded_dims, degree)
                        decompressed_data_charge = unpad_grid(decompressed_data_charge, dims)

                        compressed_data_mag = lagrange_compress_3d(mag_padded, *block_dims, degree)
                        decompressed_data_mag = lagrange_decompress_3d(compressed_data_mag, padded_dims, degree)
                        decompressed_data_mag = unpad_grid(decompressed_data_mag, dims)

                        charge_mae_val = mae(charge, decompressed_data_charge)
                        charge_mpd = mean_percentage_diff(charge, decompressed_data_charge)
                        mag_mae_val = mae(mag, decompressed_data_mag)
                        mag_mpd = mean_percentage_diff(mag, decompressed_data_mag)

                        print(f"{chgcar}, dims: {dims}, compressed dims: {compressed_data_charge.shape}, charge mae: {charge_mae_val}, charge mean_percetage_diff: {charge_mpd}, mag mae: {mag_mae_val}, mag mean_percentage_diff: {mag_mpd}")

                        if csv_writer:
                            csv_writer.writerow([
                                chgcar.name, 'lagrange',
                                dims[0], dims[1], dims[2],
                                compressed_data_charge.shape[0], compressed_data_charge.shape[1], compressed_data_charge.shape[2], compressed_data_charge.shape[3],
                                f"{charge_mae_val.item():.4e}", f"{charge_mpd.item():.4e}",
                                f"{mag_mae_val.item():.4e}", f"{mag_mpd.item():.4e}"
                            ])

                    if mode == "gaussian":
                        compressed_data_charge = gaussian_compress_3d(charge_padded, *block_dims, degree)
                        decompressed_data_charge = gaussian_decompress_3d(compressed_data_charge, padded_dims, degree)
                        decompressed_data_charge = unpad_grid(decompressed_data_charge, dims)

                        compressed_data_mag = gaussian_compress_3d(mag_padded, *block_dims, degree)
                        decompressed_data_mag = gaussian_decompress_3d(compressed_data_mag, padded_dims, degree)
                        decompressed_data_mag = unpad_grid(decompressed_data_mag, dims)

                        charge_mae_val = mae(charge, decompressed_data_charge)
                        charge_mpd = mean_percentage_diff(charge, decompressed_data_charge)
                        mag_mae_val = mae(mag, decompressed_data_mag)
                        mag_mpd = mean_percentage_diff(mag, decompressed_data_mag)

                        print(f"{chgcar}, dims: {dims}, compressed dims: {compressed_data_charge.shape}, charge mae: {charge_mae_val}, charge mean_percetage_diff: {charge_mpd}, mag mae: {mag_mae_val}, mag mean_percentage_diff: {mag_mpd}")

                        if csv_writer:
                            csv_writer.writerow([
                                chgcar.name, 'gaussian',
                                dims[0], dims[1], dims[2],
                                compressed_data_charge.shape[0], compressed_data_charge.shape[1], compressed_data_charge.shape[2], compressed_data_charge.shape[3],
                                f"{charge_mae_val.item():.4e}", f"{charge_mpd.item():.4e}",
                                f"{mag_mae_val.item():.4e}", f"{mag_mpd.item():.4e}"
                            ])

    # Close CSV file if it was opened
    if csv_file:
        csv_file.close()

if __name__ == "__main__":
    main()

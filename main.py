import click
import math
import os
import csv
import concurrent.futures
from typing import List, Tuple

from utils import *


def process_chgcar(chgcar_path, chgcar_name, mode, factor, degree, csv_writer=None):
    """Process a single CHGCAR file with the specified compression mode"""
    dims, charge, mag = parse_chgcar(chgcar_path, chgcar_name.split('.')[0] + "_no_data.txt")
    charge, mag = reshape_dims(charge, dims), reshape_dims(mag, dims)

    block_dims = [math.ceil(dim / factor) for dim in dims]
    charge_padded = pad_grid(charge, block_dims)
    mag_padded = pad_grid(mag, block_dims)
    padded_dims = charge_padded.shape

    results = {}

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

        results = {
            'filename': chgcar_name,
            'mode': 'chebyshev',
            'dims': dims,
            'compressed_dims': coeffs_charge.shape,
            'charge_mae': charge_mae_val,
            'charge_mpd': charge_mpd,
            'mag_mae': mag_mae_val,
            'mag_mpd': mag_mpd
        }

        print(f"{chgcar_name}, dims: {dims}, coeffs dims: {coeffs_charge.shape}, charge mae: {charge_mae_val}, charge mean_percentage_diff: {charge_mpd}, mag mae: {mag_mae_val}, mag mean_percentage_diff: {mag_mpd}")

    elif mode == "lagrange":
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

        results = {
            'filename': chgcar_name,
            'mode': 'lagrange',
            'dims': dims,
            'compressed_dims': compressed_data_charge.shape,
            'charge_mae': charge_mae_val,
            'charge_mpd': charge_mpd,
            'mag_mae': mag_mae_val,
            'mag_mpd': mag_mpd
        }

        print(f"{chgcar_name}, dims: {dims}, compressed dims: {compressed_data_charge.shape}, charge mae: {charge_mae_val}, charge mean_percentage_diff: {charge_mpd}, mag mae: {mag_mae_val}, mag mean_percentage_diff: {mag_mpd}")

    elif mode == "gaussian":
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

        results = {
            'filename': chgcar_name,
            'mode': 'gaussian',
            'dims': dims,
            'compressed_dims': compressed_data_charge.shape,
            'charge_mae': charge_mae_val,
            'charge_mpd': charge_mpd,
            'mag_mae': mag_mae_val,
            'mag_mpd': mag_mpd
        }

        print(f"{chgcar_name}, dims: {dims}, compressed dims: {compressed_data_charge.shape}, charge mae: {charge_mae_val}, charge mean_percentage_diff: {charge_mpd}, mag mae: {mag_mae_val}, mag mean_percentage_diff: {mag_mpd}")

    return results


@click.command()
@click.option("--factor", default=4, help="Divisor by which you wish to reduce dimensions of chgcar")
@click.option("--degree", default=3, help="The degree of the 4th dimension of data created during dimension reduction")
@click.option("--max-workers", default=None, type=int, help="Maximum number of worker processes (defaults to CPU count)")
@click.option("--mode", type=click.Choice(['chebyshev', 'lagrange', 'gaussian']), required=True)
@click.option("--output", help="Path to save results in CSV format")
@click.argument("path")
def main(factor, degree, max_workers, mode, output, path):
    """ Reduce dimensions of chgcars located in PATH by FACTOR with DEGREE """

    # Initialize CSV file if specified
    csv_file = None
    csv_writer = None

    if output:
        csv_file = open(output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'filename', 'mode',
            'dim_x', 'dim_y', 'dim_z',
            'compressed_dim_x', 'compressed_dim_y', 'compressed_dim_z', 'compressed_dim_w',
            'charge_mae', 'charge_mean_percentage_diff',
            'mag_mae', 'mag_mean_percentage_diff'
        ])

    if os.path.isdir(path):
        chgcar_files = []
        with os.scandir(path) as it:
            for entry in it:
                if ".vasp" in entry.name:
                    chgcar_files.append(entry)

        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chgcar = {
                executor.submit(process_chgcar, chgcar.path, chgcar.name, mode, factor, degree): chgcar
                for chgcar in chgcar_files
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chgcar):
                try:
                    result = future.result()
                    if result and csv_writer:
                        # Write results to CSV
                        csv_writer.writerow([
                            result['filename'], result['mode'],
                            result['dims'][0], result['dims'][1], result['dims'][2],
                            result['compressed_dims'][0], result['compressed_dims'][1],
                            result['compressed_dims'][2], result['compressed_dims'][3],
                            f"{result['charge_mae'].item():.4e}", f"{result['charge_mpd'].item():.4e}",
                            f"{result['mag_mae'].item():.4e}", f"{result['mag_mpd'].item():.4e}"
                        ])
                except Exception as exc:
                    chgcar = future_to_chgcar[future]
                    print(f"Error processing {chgcar.name}: {exc}")
    elif os.path.isfile(path):
        result = process_chgcar(chgcar.path, chgcar.name, mode, factor, degree)
        csv_writer.writerow([
            result['filename'], result['mode'],
            result['dims'][0], result['dims'][1], result['dims'][2],
            result['compressed_dims'][0], result['compressed_dims'][1],
            result['compressed_dims'][2], result['compressed_dims'][3],
            f"{result['charge_mae'].item():.4e}", f"{result['charge_mpd'].item():.4e}",
            f"{result['mag_mae'].item():.4e}", f"{result['mag_mpd'].item():.4e}"
        ])

    # Close CSV file if it was opened
    if csv_file:
        csv_file.close()

if __name__ == "__main__":
    main()

import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_single_tsv(args):
    """
    Process a single TSV file and save images and metadata.
    :param args: Tuple of TSV file path, output metadata CSV path, and output image folder
    """
    tsv_file_path, output_meta_csv, output_image_folder = args
    try:
        # Load the TSV file
        data = pd.read_csv(tsv_file_path, sep='\t', header=None)
        os.makedirs(output_image_folder, exist_ok=True)

        # Assuming the first column is the ID and the last column contains the base64 image data
        image_column_index = len(data.columns) - 1
        metadata = []

        for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {tsv_file_path}"):
            image_id = row[0]  # Assuming the first column is the ID
            image_data = row[image_column_index]  # Last column is the base64-encoded image
            other_metadata = row[1:image_column_index].to_dict()  # Collect other metadata

            try:
                # Decode the base64-encoded image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image_path = os.path.join(output_image_folder, f"{image_id}.png")
                image.save(image_path)  # Save the image

                # Add metadata entry
                metadata.append({"id": image_id, **other_metadata})
            except Exception as e:
                print(f"Error decoding or saving image for ID {image_id}: {e}")

        # Save the metadata to a CSV file
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_meta_csv, index=False)

    except Exception as e:
        print(f"Error processing TSV file {tsv_file_path}: {e}")


def process_all_tsvs(input_dir, output_dir, num_processes=None):
    """
    Process all TSV files in the input directory using multiprocessing.
    :param input_dir: Directory containing the TSV files
    :param output_dir: Directory to save the output metadata and images
    :param num_processes: Number of parallel processes to use
    """
    num_processes = num_processes or cpu_count()
    tsv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tsv')]

    tasks = []
    for tsv_file in tsv_files:
        split_name = os.path.splitext(os.path.basename(tsv_file))[0]
        output_meta_csv = os.path.join(output_dir, split_name, "metadata.csv")
        output_image_folder = os.path.join(output_dir, split_name, "images")
        tasks.append((tsv_file, output_meta_csv, output_image_folder))

    # Use multiprocessing to process TSV files with a progress bar
    with tqdm(total=len(tasks), desc="Processing all TSV files") as pbar:
        with Pool(num_processes) as pool:
            for _ in pool.imap_unordered(process_single_tsv, tasks):
                pbar.update(1)


if __name__ == "__main__":
    input_dir = "/mnt/carpedkm_data/image_gen_ds/version1/"  # Replace with your input directory containing TSV files
    output_dir = "/mnt/carpedkm_data/image_gen_ds/processed/"  # Replace with your desired output directory
    process_all_tsvs(input_dir, output_dir, num_processes=4)  # Adjust num_processes as needed
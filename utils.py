import os, re
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Set, Union


def read_bed_file(file_path: str) -> pd.DataFrame:
    """
    Reads a BED file and returns a DataFrame with 'chrom', 'start', 'end' columns.
    
    Args:
        file_path (str): Path to the BED file.
        
    Returns:
        pd.DataFrame: A DataFrame containing chromosome, start, and end positions.
    """
    try:
        with open(file_path) as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            no_header = any(ch.isdigit() for ch in first_line.split('\t'))
    except FileNotFoundError:
        logging.error(f"file {file_path} not found")
        raise FileNotFoundError(f"file {file_path} not found")

    try:
        if no_header:
            df = pd.read_csv(file_path, sep='\t', header=None)
        else:
            df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None)
    except Exception as e:
        logging.error(f"error happened when reading file {file_path}: {e}")
        raise e
    
    def is_chrom_format(s):
        return s.startswith('chr')
    
    chrom_index = None
    for i, value in enumerate(second_line.split('\t')):
        if is_chrom_format(value):
            chrom_index = i
            break
    
    if chrom_index is None:
        logging.error(f'can not recognize chrom column, please check the format of file {file_path}')
        raise ValueError(f'can not recognize chrom column, please check the format of file {file_path}')
    
    df = df.iloc[:, [chrom_index, chrom_index + 1, chrom_index + 2]]
    df.columns = ['chrom', 'start', 'end']
    df = df[df['start'] < df['end']]

    if len(df) == 0:
        logging.error(f"file {file_path} has no valid data")
        raise ValueError(f"file {file_path} has no valid data")
    
    return df

def create_binary_sequence(
    bed_df: pd.DataFrame, chrom: str, start: int, end: int, frag: int = 200
) -> np.ndarray:
    """
    Converts genomic regions into a binary sequence.

    Args:
        bed_df (pd.DataFrame): BED DataFrame with 'chrom', 'start', 'end' columns.
        chrom (str): Chromosome to process.
        start (int): Start position of the region.
        end (int): End position of the region.
        frag (int): Fragment size in base pairs (default: 200).
    
    Returns:
        np.ndarray: Binary sequence indicating overlaps.
    """
    chrom_df = bed_df[bed_df['chrom'] == chrom]

    if len(chrom_df) == 0:
        logging.error(f"no data for chromosome {chrom}")
        raise ValueError(f"")
    
    if start >= end:
        logging.error(f"start position {start} is larger than end position {end}")
        raise ValueError(f"")

    length = end - start
    num_fragments = (length + frag - 1) // frag
    expanded_length = num_fragments * frag

    if expanded_length > length:
        end = start + expanded_length
        logging.warning(f"expanded length, end adjusted to {end}, length expanded from {length} to {expanded_length}")

    binary_sequence = np.zeros(num_fragments, dtype=int)
    
    for i in range(num_fragments):
        fragment_start = start + i * frag
        fragment_end = min(fragment_start + frag, end)
        
        overlapping_peaks = chrom_df[
            (chrom_df['start'] < fragment_end) & (chrom_df['end'] > fragment_start)
        ]
        
        total_overlap_length = 0
        for _, row in overlapping_peaks.iterrows():
            overlap_start = max(row['start'], fragment_start)
            overlap_end = min(row['end'], fragment_end)
            total_overlap_length += overlap_end - overlap_start

        if total_overlap_length > frag // 2:
            binary_sequence[i] = 1
    
    return binary_sequence


def generate_multiple_sequence(histone_list: List[np.ndarray]) -> np.ndarray:
    """
    Combines multiple binary sequences into one with a unique encoding for overlaps.

    Args:
        histone_list (List[np.ndarray]): List of binary sequences for histone modifications.

    Returns:
        np.ndarray: Combined sequence with unique encodings.
    """
    histone_amount = len(histone_list)

    if histone_amount <= 1:
        logging.warning("no enough histone modification data, please add more data")

    result = histone_list[0].copy()
    for multi in range(1, histone_amount):
        new_histone = histone_list[multi]
        shape_result = result.shape
        shape_new = new_histone.shape
        if shape_new != shape_result:
            logging.warning(f'the shape of histone {multi} is {shape_new}, the shape of histone 0 is {shape_result}, please check the data')
            raise ValueError(f'the shape of histone {multi} is {shape_new}, the shape of histone 0 is {shape_result}, please check the data')
        result += new_histone * (10 ** multi)

    return result


def map_observations(observations: np.ndarray, mods: int = 4) -> np.ndarray:
    """
    Maps raw observations to encoded values based on binary representation.

    Args:
        observations (np.ndarray): Observations to map.
        mods (int): Number of modifications (default: 4).
    
    Returns:
        np.ndarray: Mapped observations.
    """
    obs_map = {int(bin(i)[2:]):i for i in range(2 ** mods)}
    return np.array([obs_map[o] for _, o in np.ndenumerate(observations)])


def modifications_to_binary(records: List[str], modifications: List[str]) -> int:
    """
    Converts a list of modifications into a binary-encoded integer.

    Args:
        records (List[str]): List of modification names.

    Returns:
        int: Binary-encoded representation of modifications.
    """
    modification_map = {mod: bit_position for mod, bit_position in zip(modifications, range(len(modifications)))}
    binary_result = 0
    
    for record in records:
        for mod, bit_position in modification_map.items():
            if mod in record:
                binary_result |= (1 << bit_position)
    
    return binary_result


def read_all_bed_file(
    chip_dir: str, chrom: str, start: int, end: int, frag: int = 200
) -> tuple[np.ndarray, list[str]]:
    """
    Reads all BED files in a directory,
    converts them into binary sequences, and aggregates them into a single array.

    Args:
        chip_dir (str): Directory containing the BED files.
        chrom (str): The chromosome to filter the data by (e.g., "chr1").
        start (int): The start coordinate for the region of interest.
        end (int): The end coordinate for the region of interest.
        frag (int, optional): Fragment size for dividing the region. Defaults to 200.

    Returns:
        tuple[np.ndarray, list[str]]:
            - A NumPy array of binary sequences for each histone modification.
            - A list of filenames corresponding to the processed BED files.

    Raises:
        ValueError: If no valid BED files are found in the directory.
    """
    result = []
    records = []
    mods = set()

    for filename in sorted(os.listdir(chip_dir)):
        if filename.endswith('.bed'):
            mods.add([part for part in re.split('[._]', filename) if part][0])
            file_path = os.path.join(chip_dir, filename)

            data = read_bed_file(file_path)
            data = create_binary_sequence(data, chrom, start, end, frag)
            result.append(data)
            records.append(filename)

    if len(records) == 0:
        logging.error(f"No valid bed file found in {chip_dir}")
        raise ValueError(f"No valid bed file found in {chip_dir}")

    logging.info(f"Read the following histone modifications: {', '.join(sorted(mods))}")
    return np.array(result), mods


def sequence_to_bed(sequence: np.ndarray, chrom: str, start: int) -> pd.DataFrame:
    """
    Converts a binary sequence into a BED format DataFrame.

    Args:
        sequence (np.ndarray): A binary sequence where 1 indicates a peak and 0 indicates no peak.
        chrom (str): Chromosome name for the BED file (e.g., "chr1").
        start (int): The start position of the sequence region.

    Returns:
        pd.DataFrame: A DataFrame in BED format with columns ['chrom', 'start', 'end'].

    Raises:
        ValueError: If the sequence contains invalid values (not 0 or 1).
    """
    bed_entries = []
    
    peak_start = False
    for index, value in enumerate(sequence):
        if value != 1 and value != 0:
            logging.error(f'the {index}th data {value} in the sequence is not valid')
            raise ValueError(f'the {index}th data {value} in the sequence is not valid')
        elif value == 1:
            if not peak_start:
                peak_start = True
                start_position = start + index * 200
        else:
            if peak_start:
                end_position = start + index * 200
                bed_entries.append([chrom, start_position, end_position])
                peak_start = False

    if peak_start:
        end_position = start + len(sequence) * 200
        bed_entries.append([chrom, start_position, end_position])

    bed_df = pd.DataFrame(bed_entries, columns=['chrom', 'start', 'end'])
    
    return bed_df


def calculate_predicted_probs(hmm, test_observation: np.ndarray) -> np.ndarray:
    """
    Computes posterior probabilities of the hidden states for a given test observation
    using a Hidden Markov Model (HMM).

    Args:
        hmm: An HMM instance with methods `forward_log` and `backward_log`.
        test_observation (np.ndarray): An array of observed states for testing.

    Returns:
        np.ndarray: Posterior probabilities for the hidden states, focusing on the "open" state.
    """
    _, alpha = hmm.forward_log(test_observation)
    beta = hmm.backward_log(test_observation)

    log_posterior = alpha + beta
    log_posterior -= np.logaddexp.reduce(log_posterior, axis=0, keepdims=True)
    posterior_probs = np.exp(log_posterior)
    return posterior_probs[1, :]


if __name__ == '__main__':
    pass

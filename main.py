import os
import logging
import argparse

from model import *
from utils import *
from visualization import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--bed', help='ChIP-seq test file path', required=True)
    parser.add_argument('-o', '--output', help='output path', required=True)
    parser.add_argument('-c', '--chrom', help='choose a chromosome', required=True)
    parser.add_argument('-t', '--train', help='ChIP-seq train file path')
    parser.add_argument('-s', '--start', help='chromosome start position', type=int)
    parser.add_argument('-e', '--end', help='chromosome end position', type=int)
    parser.add_argument('-a', '--atac', help='ATAC-seq file path')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    root_path = os.path.dirname(os.path.abspath(__file__))

    chrom = 'chr' + args.chrom
    start = args.start if args.start else 0
    end = args.end if args.end else 100000

    train_dir = args.train if args.train else os.path.join(root_path, 'Chip-seq/train/')
    train_data, train_histone_names = read_all_bed_file(train_dir, chrom, start, end)
    train_data = generate_multiple_sequence(train_data)
    train_observation = map_observations(train_data).reshape(1, -1)

    transition = np.array([[0.4, 0.6], [0.4, 0.6]])
    emission = np.array([[1 / 16] * 16, [1 / 16] * 16])
    initial = np.array([[0.5, 0.5]])
    log_transition = np.log(transition)
    log_emission = np.log(emission)
    log_initial = np.log(initial)
    hmm = HMM(2, 16, log_transition, log_emission, log_initial)

    hmm.baum_welch_log(train_observation, 500)
    
    test_dir = args.bed
    test_data, test_histone_names = read_all_bed_file(test_dir, chrom, start, end)
    test_data = generate_multiple_sequence(test_data)
    test_observation = map_observations(test_data).reshape(1, -1)

    test_mods = bin(modifications_to_binary(test_histone_names))[2:]
    hmm.adjust_emission_matrix(test_mods)

    path, path_prob = hmm.viterbi_log(test_observation)
    if hmm.log_initial[0] < hmm.log_initial[1]:
        path = -path + 1
    sequence_to_bed(path, chrom, start).to_csv(os.path.join(args.output, 'predicted.bed'), sep='\t', index=False, header=False)


    if args.atac:
        atac_filepath = args.atac
        atac = read_bed_file(atac_filepath)
        atac = create_binary_sequence(atac, chrom, start, end)
        predicted_probs = calculate_predicted_probs(hmm, test_observation)
        if hmm.log_initial[0] < hmm.log_initial[1]:
            predicted_probs = 1 - predicted_probs
        draw_roc(atac.tolist(), predicted_probs)

    # draw_tracks(os.path.join(args.output, 'predicted.bed'), atac_filepath, args.output, chrom, start, end)

if __name__ == '__main__':
    main()


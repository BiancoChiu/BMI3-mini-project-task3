# BMI3-mini-project-task3
Task: Chromatin state predictor


Developed By: DENG Yanqi, Duan Wenzhuo, GU Chengbin, SHEN Yu, ZHAO Bingkang


---

## Command-line Arguments
| Argument                | Description                                                   | Required | Example                     |
|-------------------------|---------------------------------------------------------------|----------|-----------------------------|
| `-b`, `--bed`           | Path to the ChIP-seq test file.                               | Yes      | `data/test.bed`            |
| `-o`, `--output`        | Path to save the output results.                              | Yes      | `results/`                 |
| `-c`, `--chrom`         | Chromosome to analyze.                                        | Yes      | `1`                        |
| `-t`, `--train`         | Path to additional ChIP-seq files for training.               | No       | `data/train_extra.bed`     |
| `-s`, `--start`         | Start position of the chromosome (default: `0`).              | No       | `100000`                   |
| `-e`, `--end`           | End position of the chromosome (default: `100000`).           | No       | `200000`                   |
| `-a`, `--atac`          | Path to the ATAC-seq file for evaluation.                     | No       | `data/atac_seq.bed`        |

## Example Command
```python
python main.py \
    -b ChIP-seq/test \
    -o output/ \
    -c 1 \
    -t ChIP-seq/train \
    -s 100000 \
    -e 200000 \
    -a ATAC-seq/Sample_0382.bed
```

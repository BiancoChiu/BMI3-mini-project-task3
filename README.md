# BMI3-mini-project-task3
Task: Chromatin state predictor


Developed By: DENG Yanqi, DUAN Wenzhuo, GU Chengbin, SHEN Yu, ZHAO Bingkang
<img width="1212" alt="chr1" src="https://github.com/user-attachments/assets/fe4390d3-2dbb-446c-9ab8-1a7bac6e36ec" />


---

## Command-line Arguments
| Argument                | Description                                                   | Required | Example                     |
|-------------------------|---------------------------------------------------------------|----------|-----------------------------|
| `-b`, `--bed`           | Path to the ChIP-seq test files.                               | Yes      | `ChIP-seq/test`            |
| `-o`, `--output`        | Path to save the output results.                              | Yes      | `output/`                 |
| `-c`, `--chrom`         | Chromosome to analyze.                                        | Yes      | `1`                        |
| `-t`, `--train`         | Path to ChIP-seq files for training.               | No       | `ChIP-seq/train`     |
| `-s`, `--start`         | Start position of the chromosome (default: `0`).              | No       | `900000`                   |
| `-e`, `--end`           | End position of the chromosome (default: `100000`).           | No       | `980000`                   |
| `-a`, `--atac`          | Path to the ATAC-seq file for evaluation.                     | No       | `ATAC-seq/GM12878_ATAC.bed`        |

## Example Command

```python
python main.py \
    -b ChIP-seq/test \
    -o output/ \
    -c 1 \
    -t ChIP-seq/train \
    -s 900000 \
    -e 980000 \
    -a ATAC-seq/GM12878_ATAC.bed

# optional: start from gui
python gui.py
```

## Structure
```
.
├── README.md
├── __init__.py
├── gui.py
├── main.py
├── model.py
├── utils.py
├── visualization.py
├── ATAC-seq
│   └── ...
├── ChIP-seq
│   ├── extra
│   │   └── ...
│   ├── test
│   │   └── ...
│   └── train
│       └── ...
├── assets
│   └── ...
├── output
│   └── ...
└── requirements.txt
```

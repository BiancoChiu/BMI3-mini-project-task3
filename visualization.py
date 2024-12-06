import os
import matplotlib.pyplot as plt 
from PIL import Image
from sklearn.metrics import roc_curve, auc
from typing import List, Union


def draw_roc(true: List[int], score: List[float]) -> None:
    """
    Plot and display the ROC curve based on true labels and predicted scores.

    Args:
        true (List[int]): List of true binary labels (0 or 1).
        score (List[float]): List of predicted scores or probabilities for the positive class.
    """
    fpr, tpr, thresholds = roc_curve(true, score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def save_roc(true: List[int], score: List[float], save_path: str) -> None:
    """
    Plot the ROC curve and save it as an image file.

    Args:
        true (List[int]): List of true binary labels (0 or 1).
        score (List[float]): List of predicted scores or probabilities for the positive class.
        save_path (str): Path to save the ROC curve plot image.
    """
    fpr, tpr, thresholds = roc_curve(true, score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    plt.savefig(save_path, transparent=True)
    plt.close() 


def draw_tracks(predicted_path: str, atac_path: str, save_path: str, 
                chrom: str, start: int, end: int) -> None:
    """
    Generate a visualization of genomic tracks and save the image.

    Args:
        predicted_path (str): Path to the file containing predicted tracks.
        atac_path (str): Path to the file containing ATAC-seq data tracks.
        save_path (str): Directory where the output files and image will be saved.
        chrom (str): Chromosome name (e.g., "chr1").
        start (int): Start position of the genomic region.
        end (int): End position of the genomic region.
    """
    os.system(f"make_tracks_file --trackFiles {predicted_path} {atac_path} -o {os.path.join(save_path, 'tracks.ini')}")
    os.system(f"pyGenomeTracks --tracks {os.path.join(save_path, 'tracks.ini')} --region {chrom}:{start}-{end} --outFileName {os.path.join(save_path, 'nice_image.png')}")


def to_transparent_background(image_path: str) -> None:
    """
    Convert white background in an image to transparent.

    Args:
        image_path (str): Path to the input image file.
    """
    img = Image.open(image_path)
    img = img.convert("RGBA")
    data = img.getdata()
    new_data = []
    for item in data:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(image_path, 'PNG')

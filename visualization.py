import os
import matplotlib.pyplot as plt 
from PIL import Image
from sklearn.metrics import roc_curve, auc

def draw_roc(true, score):
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


def save_roc(true, score, save_path):
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


def draw_tracks(predicted_path, atac_path, save_path, chrom, start, end):
    os.system(f"make_tracks_file --trackFiles {predicted_path} {atac_path} -o {os.path.join(save_path, 'tracks.ini')}")
    os.system(f"pyGenomeTracks --tracks {os.path.join(save_path, 'tracks.ini')} --region {chrom}:{start}-{end} --outFileName {os.path.join(save_path, 'nice_image.png')}")


def to_transparent_background(image_path):
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

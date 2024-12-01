
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

import os
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label, filedialog
from PIL import Image, ImageTk

from model import *
from utils import *
from visualization import *

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(ROOT_PATH, 'output')
ASSETS_PATH = os.path.join(ROOT_PATH, 'assets/frame0')
TRAIN_PATH = None
TEST_PATH = None
# TRAIN_PATH = os.path.join(ROOT_PATH, 'Chip-seq/train/')
# TEST_PATH = os.path.join(ROOT_PATH, 'Chip-seq/test/')
ATAC_PATH = os.path.join(ROOT_PATH, 'ATAC-seq/merged_ATAC.bed')


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def select_training_path():
    global TRAIN_PATH
    file_path = filedialog.askdirectory(
        title="Select ChIP-seq BED directory for training",
    )
    entry_1.insert("end", f"ChIP-seq training path: {file_path}\n")
    TRAIN_PATH = file_path


def select_testing_path():
    global TEST_PATH
    file_path = filedialog.askdirectory(
        title="Select ChIP-seq BED directory for testing",
    )
    entry_1.insert("end", f"ChIP-seq testing path: {file_path}\n")
    TEST_PATH = file_path


def display_image_on_label(label, image_path, margin=10):
    """
    Load and display an image on the given label with a margin around it.
    Args:
        label: The Label widget where the image will be displayed.
        image_path: The path to the image file.
        margin: The margin size (in pixels) around the image.
    """
    try:
        label_width = int(label.winfo_width())
        label_height = int(label.winfo_height())
        
        img_width = label_width - int(0.5 * margin)
        img_height = label_height - 2 * margin
        
        img = Image.open(image_path)
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)

        label.img = img_tk
        label.config(image=img_tk)
        entry_1.insert("end", f"Image displayed on label with margin: {image_path}\n")
    except Exception as e:
        entry_1.insert("end", f"Error displaying image: {e}\n")


def handle_main_click():
    """
    Function to handle the logic when button_1 is clicked.
    """
    entry_1.delete("1.0", "end")

    if TEST_PATH is None:
        entry_1.insert("end", "Error: Testing path is not selected.\n")
        return
    
    chrom = 'chr' + entry_3.get().strip()
    start = int(entry_4.get().strip())
    end = int(entry_5.get().strip())
    
    entry_1.insert("end", "Checking input fields...\n")
    if not chrom:
        entry_1.insert("end", "Error: Chrom is empty.\n")
        return
    if start is None or start == "":
        entry_1.insert("end", "Error: Start is empty.\n")
        return
    if not end:
        entry_1.insert("end", "Error: End is empty.\n")
        return
    
    entry_1.insert("end", f"Chrom: {chrom}\n")
    entry_1.insert("end", f"Start Position: {start}\n")
    entry_1.insert("end", f"End Position: {end}\n")

    entry_1.insert("end", f"")
    entry_1.insert("end", f"Loding available training data...\n")
    train_data, train_histone_names = read_all_bed_file(os.path.join(ROOT_PATH, 'Chip-seq/train/'), chrom, start, end)
    train_data = generate_multiple_sequence(train_data)
    train_observation = map_observations(train_data).reshape(1, -1)

    if TRAIN_PATH:
        extra_data, extra_histone_names = read_all_bed_file(TRAIN_PATH, chrom, start, end)
        entry_7.insert("end", f"Extra data:\n{extra_histone_names}\n")

        extra_data = generate_multiple_sequence(extra_data)
        extra_observation = map_observations(extra_data).reshape(1, -1)
        train_observation = np.concatenate((train_observation, extra_observation), axis=1)
        print(train_observation.shape)
    entry_1.insert("end", f"Training data loaded successfully.\n")

    transition = np.array([[0.6, 0.4], [0.6, 0.4]])
    emission = np.array([[1 / 16] * 16, [1 / 16] * 16])
    initial = np.array([[0.5, 0.5]])
    log_transition = np.log(transition)
    log_emission = np.log(emission)
    log_initial = np.log(initial)
    hmm = HMM(2, 16, log_transition, log_emission, log_initial)
    entry_1.insert("end", f"Training HMM model...\n")
    hmm.baum_welch_log(train_observation, 500)

    entry_1.insert("end", f"Loading testing data...\n")
    test_data, test_histone_names = read_all_bed_file(TEST_PATH, chrom, start, end)
    entry_7.insert("end", f"Testing data:\n{test_histone_names}\n")
    test_data = generate_multiple_sequence(test_data)
    test_observation = map_observations(test_data).reshape(1, -1)

    test_mods = bin(modifications_to_binary(test_histone_names))[2:]
    hmm.adjust_emission_matrix(test_mods)

    path, path_prob = hmm.viterbi_log(test_observation)
    if hmm.log_initial[0] < hmm.log_initial[1]:
        path = -path + 1
    sequence_to_bed(path, chrom, start).to_csv(os.path.join(OUTPUT_PATH, 'predicted_accessibility.bed'), sep='\t', index=False, header=False)

    atac = read_bed_file(ATAC_PATH)
    atac = create_binary_sequence(atac, chrom, start, end)
    predicted_probs = calculate_predicted_probs(hmm, test_observation)
    if hmm.log_initial[0] < hmm.log_initial[1]:
        predicted_probs = 1 - predicted_probs
    save_roc(atac.tolist(), predicted_probs, os.path.join(OUTPUT_PATH, 'roc_curve.png'))

    display_image_on_label(entry_6, os.path.join(OUTPUT_PATH, 'roc_curve.png'))
    entry_1.insert("end", f"ROC curve saved successfully.\n")
    draw_tracks(os.path.join(OUTPUT_PATH, 'predicted_accessibility.bed'), ATAC_PATH, OUTPUT_PATH, chrom, start, end)
    to_transparent_background(os.path.join(OUTPUT_PATH, 'nice_image.png'))
    display_image_on_label(entry_2, os.path.join(OUTPUT_PATH, 'nice_image.png'))
    entry_1.insert("end", f"Tracks image saved successfully.\n")

window = Tk()

window.geometry("1212x842")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 842,
    width = 1212,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    153.0,
    305.0,
    image=image_image_1
)

canvas.create_text(
    483.0,
    29.0,
    anchor="nw",
    text="ChromAccHMM",
    fill="#000000",
    font=("Inter Bold", 32 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=113.0,
    y=432.0,
    width=80.0,
    height=40.0
)

canvas.create_text(
    43.0,
    165.0,
    anchor="nw",
    text="ChIP-seq file path to train",
    fill="#000000",
    font=("Inter Bold", 15 * -1)
)

canvas.create_text(
    43.0,
    212.0,
    anchor="nw",
    text="ChIP-seq file path to test",
    fill="#000000",
    font=("Inter Bold", 15 * -1)
)

canvas.create_text(
    584.0,
    810.0,
    anchor="nw",
    text="Developed By: DENG Yanqi, DUAN Wenzhuo, GU Chengbin, SHEN Yu, ZHAO Bingkang",
    fill="#000000",
    font=("Inter", 15 * -1)
)

canvas.create_text(
    42.0,
    810.0,
    anchor="nw",
    text="BMI-mini project task 3: Chromatin state predictor",
    fill="#000000",
    font=("Inter", 15 * -1)
)

canvas.create_text(
    823.0,
    522.0,
    anchor="nw",
    text="Log Viewer",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    401.0,
    522.0,
    anchor="nw",
    text="AUC",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    93.0,
    522.0,
    anchor="nw",
    text="Loaded Data",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    686.0,
    70.0,
    anchor="nw",
    text="Visualization",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=241.0,
    y=160.999995931983,
    width=24.41574217379093,
    height=24.41574217379093
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=241.0,
    y=208.0,
    width=24.0,
    height=24.0
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    876.5,
    678.0,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#ECE6F0",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=593.0,
    y=559.0,
    width=567.0,
    height=236.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    747.0,
    305.0,
    image=entry_image_2
)
entry_2 = Label(
    bd=0,
    bg="#ECE6F0",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=334.0,
    y=109.0,
    width=826.0,
    height=390.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    92.0,
    272.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    92.0,
    328.0,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    92.0,
    384.0,
    image=image_image_4
)

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    198.0,
    272.0,
    image=entry_image_3
)
entry_3 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=144.0,
    y=248.0,
    width=108.0,
    height=46.0
)

entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    198.0,
    328.0,
    image=entry_image_4
)
entry_4 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_4.place(
    x=144.0,
    y=304.0,
    width=108.0,
    height=46.0
)

entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    198.0,
    384.0,
    image=entry_image_5
)
entry_5 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_5.place(
    x=144.0,
    y=360.0,
    width=108.0,
    height=46.0
)

entry_image_6 = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_6 = canvas.create_image(
    423.0,
    678.0,
    image=entry_image_6
)
entry_6 = Label(
    bd=0,
    bg="#ECE6F0",
    fg="#000716",
    highlightthickness=0
)
entry_6.place(
    x=334.0,
    y=559.0,
    width=178.0,
    height=236.0
)

entry_image_7 = PhotoImage(
    file=relative_to_assets("entry_7.png"))
entry_bg_7 = canvas.create_image(
    152.0,
    678.0,
    image=entry_image_7
)
entry_7 = Text(
    bd=0,
    bg="#ECE6F0",
    fg="#000716",
    highlightthickness=0
)
entry_7.place(
    x=52.0,
    y=559.0,
    width=200.0,
    height=236.0
)

button_1.config(command=handle_main_click)
button_2.config(command=select_training_path)
button_3.config(command=select_testing_path)

window.resizable(False, False)
window.mainloop()

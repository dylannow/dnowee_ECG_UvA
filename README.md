# Detecting Chagas Disease from 12-Lead ECGs Using Deep Learning

**Bachelor Thesis AI – University of Amsterdam**  
**Author:** Dylan Nowee  
**Date:** June 2025

This repository contains all code, documentation, and resources developed for my bachelor thesis on **automated detection of Chagas disease from 12-lead electrocardiograms (ECGs)** using **convolutional neural networks (CNNs)** and **wavelet-based preprocessing**.

---

## Overview

Chagas disease is a parasitic infection that can cause chronic cardiac complications. Early detection from ECGs is challenging due to the subtle changes in ECG signals.  
This research investigates how **deep learning architectures**, trained on **wavelet-transformed ECGs**, can improve diagnostic performance and interpretability.

The project focused on:
- Implementing a **preprocessing pipeline** for 12-lead ECGs
- Applying a **Morlet continuous wavelet transform (CWT)** to represent ECGs in the time–frequency domain
- Training and evaluating multiple **2D CNN** classifiers on balanced datasets derived from the **CODE-15%** source
- Optimizing the workflow for execution on the **Snellius HPC** infrastructure

---

## Repository Structure
├── bachelor_thesis_dnowee.pdf # Final bachelor thesis document (uploaded soon...) <br />
├── README.md # Project documentation <br />
├── requirements.txt # Python dependencies <br />
├── data/ # data from https://zenodo.org/records/4916206 <br />
│ ├── chagas_class_balance.png # the balance of CODE-15 data visualised <br />
│ ├── data_before_balancing.png # the balance of CODE-15 data visualised <br />
├── code/ <br />
│ ├── models/ <br />
│ │ ├── 2d_cnn v1 (efficientnet based)/ <br />
| | |   └── team_code.py # code for creating, training and running 2d cnn v1 <br />
│ │ └── 2d_cnn v2 (efficientnet l-based)/ <br />
| | |   └── team_code.py # code for creating, training and running 2d cnn v2 <br />
│ ├── prepare_code15_data.py # Custom PyTorch dataset for CODE-15 ECGs <br />
│ ├── preprocess_code15_wavelets.py # Morlet wavelet transform & filtering pipeline <br />
| ├── train_model.py # functions for training the model for challenge <br />
| ├── run_model.py # functions for running the model for challenge <br />
| ├── evaluate_model_more_metrics.py # functions for evaluating the model <br />
| └── make_npy_array_calibration_preds.py # script for creating temporary GOTOs in working with limited space on snellius server <br />
| ├── calibrate_model.py # functions for calibrating the model <br />
| ├── helper_code.py # various helper functions defined for the challenge <br />
├── results/ <br />
│ ├── pr_curve_v1.png <br />
│ ├── pr_curve_v2.png <br />
│ ├── roc_curve_v1.png <br />
│ ├── roc_curve_v2.png <br />
│ └── pytorch dataset visualisation/ <br />
│ ├── sample1_ecg_hp_0.5_filter.png # an ecg sample with high pass filter <br />
│ └── sample1_preprocessed_cwt.png # an ecg sample preprocessed to the time-frequency domain using a continuous wavelet transform <br />
└── slurm/ <br />
├── preprocessing_morlet_hp_znorm_4096.job # Batch job for preprocessing on Snellius <br />
└── train_model_balanced.job # Batch job for model training to be altered manually <br />


---

## Preprocessing Pipeline

The **ECG preprocessing workflow** converts raw 12-lead signals into time–frequency representations suitable for CNN input.

**Main steps:**
1. **Signal Loading:**  
   Read `.dat` / `.hea` files from the WFDB format using `wfdb.rdrecord()`.
2. **High-Pass Filtering:**  
   Remove baseline wander (cutoff = 0.5 Hz).
3. **Z-score Normalization:**  
   Normalize each lead independently to zero mean and unit variance.
4. **Zero-Padding:**  
   Pad signals to 4096 samples (≈10.2 seconds at 400 Hz).
5. **Morlet Continuous Wavelet Transform:**  
   Apply the complex Morlet CWT using `neuroDSP.timefrequency.wavelet_transform` with logarithmically spaced frequencies.
6. **Power Spectrum Conversion:**  
   Compute the magnitude squared of the wavelet coefficients to obtain a spectrogram-like 2D representation.
7. **Stacking Leads:**  
   Combine all 12-lead spectrograms into a multi-channel tensor for CNN input.

**Preprocessing output format:**  
Each ECG sample → tensor of shape **[12, n_frequencies, n_timepoints]**

---

## Model Overview

A **2D Convolutional Neural Network (CNN)** was trained on wavelet spectrograms to classify ECGs as *Chagas-positive* or *Chagas-negative*.  

**Key design choices:**
- 2D convolution layers with ReLU activations  
- Batch normalization and dropout for stability  
- Global average pooling before the final classification layer  
- Binary cross-entropy loss with balanced class sampling

**Training details:**
- Framework: PyTorch 2.7  
- Optimizer: Adam (lr = 1e-4)  
- Batch size: 16  
- Epochs: 5  
- Balanced dataset (50/50 positive vs. negative)  

---

## Data Description

- **CODE-15% Dataset** – 15% subset of the Brazilian CODE ECG collection  
- **Sampling rate:** 400 Hz  
- **Lead configuration:** Standard 12-lead ECG (I, II, III, aVR, aVL, aVF, V1–V6)

*Raw CODE-15% dataset is publicly available at https://zenodo.org/records/4916206.*  

---

## Running the Code

### 1. Setup
```bash
git clone https://github.com/dylannow/dnowee_ECG_UvA
cd dnowee_ECG_UvA
pip install -r requirements.txt
```
---

### 2. Creating data for these scripts

These instructions use `code15_input` as the path for the input data files and `code15_output` for the output data files, but you can replace them with the absolute or relative paths for the files on your machine.

1. Download and unzip one or more of the `exam_part` files and the `exams.csv` file in the [CODE-15% dataset](https://zenodo.org/records/4916206).

2. Download and unzip the Chagas labels, i.e., the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip) file from the PhysioNet website.

3. Convert the CODE-15% dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

        python prepare_code15_data.py \
            -i code15_input/exams_part0.hdf5 code15_input/exams_part1.hdf5 \
            -d code15_input/exams.csv \
            -l code15_input/code15_chagas_labels.csv \
            -o code15_output/exams_part0 code15_output/exams_part1

Each `exam_part` file in the [CODE-15% dataset](https://zenodo.org/records/4916206) contains approximately 20,000 ECG recordings. You can include more or fewer of these files to increase or decrease the number of ECG recordings, respectively. 

*[Challenge website](https://physionetchallenges.org/2025/)* 

--- 
## HPC Execution (Snellius)
Training and preprocessing were executed on the Snellius HPC cluster using SLURM batch scripts.
Each array job allocated 8 CPU cores, 1 GPU, and 64 GB RAM per task.
Preprocessing (wavelet transform) was CPU-based; CNN training ran on GPU.

---

## Results
- Best Model: 2D CNN v1 
- Accuracy: 0.75 
- AUC: 0.83 
- recall: 0.74 

Visualizations of example wavelet transforms and model statistics are available in `results/pytorch dataset visualisation`, `results/pr_curve_v1.png` and `results/roc_curve_v1.png`.

---

## Thesis document
The full thesis document will be uploaded in this repository under `bachelor_thesis_dnowee.pdf`.

---

## Acknowledgements
This work was conducted as part of the Bachelor of Science in Artificial Intelligence at the University of Amsterdam (UvA).
Special thanks to Dr. Navchetan Awasthi for supervision and feedback, 
and to PhysioNet for providing a base for the code and the opurtunity to work on this project.

---

## Contact
For questions or collaboration:
- Dylan Nowee
- dylan.nowee@gmail.com

---

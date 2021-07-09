# ReSGAN-ICH

This repository contains code for learning semantic manipulation of head CT-scans as described in the paper ReSGAN: Intracranial Hemorrhage Segmentation with Residuals of Synthetic Brain CT Scans. 

## How to Run

1. Install the required libraries, list below contains the important packages.
2. Get dataset from https://physionet.org/content/ct-ich/1.3.1/. The data should have '[dataset root]/images' and '[dataset root]/label' folders with images after processing.
3. Enter '[dataset root]' to 'create_ctich_split.py', 'experiment/train.sh', 'experiment/test.sh'. 
4. Run 'create_ctich_split.py' and create a '[dataset root]/splits' folder with the resulting files.
7. Navigate the to code root folder and run 'bash experiment/train.sh' to train the model.
8. Navigate the to code root folder and run 'bash experiment/test.sh' to create dataset of synthetic CT-scans with hemorrhage removed.

## Packages Used

- numpy==1.16.6
- opencv-python==4.2.0.32
- Pillow==6.2.2
- scikit-learn==0.22.1
- scikit-image==0.14.5
- scipy==1.2.3
- tensorboard==1.14.0
- tensorboard-logger==0.1.0
- torch==1.5.0
- torchvision==0.5.0

## Limitations

Please note that this code is developed to be run on my system and cannot be guaranteed to work on everywhere, so make sure to use a virtual environment to prevent any issues at first. This code can be used for semantic manipulation of CT scans, the code for pre-processing (registration and skull-stripping), as well as training segmentation networks can not be included as they are not open and not written by me. The default settings in this code may differ from the ones used in the paper.

The current version is quickly adapted by removing private components of code. I will verify the package versions, default settings and confirm functionality of this code at later time. 

## Credit

Inspiration is taken and some parts of the code are borrowed from repositories below.
- https://github.com/NVlabs/SPADE
- https://github.com/tzt101/CLADE
- https://github.com/taesungp/contrastive-unpaired-translation.

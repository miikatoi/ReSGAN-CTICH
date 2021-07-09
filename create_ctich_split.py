import pandas as pd
import numpy as np
import os

dataroot = ''
annotation_filename = 'hemorrhage_diagnosis_raw_ct.csv'
image_path = 'data/image'
label_path = 'data/label'
ignore_patientlist = [70]   # patient 70 is not ICH, it is better to leave out

# 5 splits for cross validation, 1 for testing
splits_list = [1, 2, 3, 4, 5, 6]
splits = [[], [], [], [], [], []]


def get_paths(df, patientlist):
    paths, labels, slices = [], [], []
    for patientnum in patientlist:
        for slicenum in df[df.PatientNumber == patientnum].SliceNumber:
            index = df[(df.PatientNumber == patientnum) & (df.SliceNumber == slicenum)].index[0]
            paths.append(os.path.join(image_path, '{}.png'.format(index)))
            labels.append(os.path.join(label_path, '{}.png'.format(index)))
            slices.append('{}'.format(slicenum))
    return paths, labels, slices


def save_to_file(fpath, paths, labels, slices):
    with open(fpath, 'w') as f:
        for img_path, label_path, slicenum in zip(paths, labels, slices):
            f.write(img_path + '\t' + label_path + '\t' + slicenum + '\n')


# read csv files and couple with id and split id corresponding to the filename
df = pd.read_csv(os.path.join(dataroot, annotation_filename))

# filter
labels = df
#labels = df[df.No_Hemorrhage == 0]     # uncomment to remove normal samples from split
for patientnumber in ignore_patientlist:
    labels = labels[labels.PatientNumber != patientnumber]

# order of patients wtr number or ICH slices
hemorrhage_counts = labels.PatientNumber.value_counts()
ordered_patients = hemorrhage_counts.index

# circular enumeration
for idx in range(len(ordered_patients)):
    list_idx = idx % len(splits_list)
    splits[list_idx].append(ordered_patients[idx])

# create test
patientlist = splits[0]
fname = 'test_split.lst'
save_to_file(fname, *get_paths(labels, patientlist))

# create train/val
patientlists = splits[1:]
for split_idx in range(len(patientlists)):

    trainlist = []
    for idx, lst in enumerate(patientlists):
        if idx != split_idx:
            trainlist = trainlist + lst
    vallist = patientlists[split_idx]

    fname = 'cv{}_val_split.lst'.format(split_idx + 1)
    save_to_file(fname, *get_paths(labels, vallist))

    fname = 'cv{}_train_split.lst'.format(split_idx + 1)
    save_to_file(fname, *get_paths(labels, trainlist))


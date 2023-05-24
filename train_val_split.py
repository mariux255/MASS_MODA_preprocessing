import os
import random
import shutil

root_path = '/scratch/s174411/MM_C1'

def remove_duplicates(l):
    seen = {}
    res = []
    for item in l:
        if item in seen:
            res.append(item)
        else:
            seen[item] = 1
    return res

def train_val_fun(root_path, val_proportion = 0.2):
    processed_data_dir = root_path + '/1D_MASS_MODA_processed'
    input_path = processed_data_dir + '/input/'
    labels_path = processed_data_dir + '/labels/'
    train_path = root_path + '/TRAIN/'
    val_path = root_path + '/VAL/'
    train_input_path = root_path + '/TRAIN/input/'
    train_label_path = root_path + '/TRAIN/labels/'
    val_input_path = root_path + '/VAL/input/'
    val_label_path = root_path + '/VAL/labels/'

    dirs_list = []
    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            if dir[0] != '0':
                continue 
            dirs_list.append(dir)
    
    val_size = int(len(dirs_list)*val_proportion)

    train_dirs_list = dirs_list.copy()
    val_dirs_list = []
    for i in range(val_size):
        seq_choice = random.choice(train_dirs_list)
        val_dirs_list.append(seq_choice)
        train_dirs_list.remove(seq_choice)

    dupli = remove_duplicates(train_dirs_list)
    print(dupli)

    

    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    if not os.path.isdir(val_path):
        os.mkdir(val_path)

    if not os.path.isdir(train_input_path):
        os.mkdir(train_input_path)

    if not os.path.isdir(train_label_path):
        os.mkdir(train_label_path)

    if not os.path.isdir(val_input_path):
        os.mkdir(val_input_path)

    if not os.path.isdir(val_label_path):
        os.mkdir(val_label_path)

    for seq in train_dirs_list:
        dest_input = os.path.join(train_input_path,seq)
        dest_labels = os.path.join(train_label_path,seq)
        shutil.copytree(os.path.join(input_path,seq), dest_input)
        shutil.copytree(os.path.join(labels_path,seq), dest_labels)

    for seq in val_dirs_list:
        dest_input = os.path.join(val_input_path,seq)
        dest_labels = os.path.join(val_label_path,seq)
        shutil.copytree(os.path.join(input_path,seq), dest_input)
        shutil.copytree(os.path.join(labels_path,seq), dest_labels)

train_val_fun(root_path=root_path)
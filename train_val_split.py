import os
import random
import shutil

root_path = '/home/marius/Documents/THESIS/data'

def train_val_fun(root_path, val_proportion = 0.2):
    processed_data_dir = root_path + '/1D_MASS_MODA_processed'
    dirs_list = []
    for root, dirs, files in os.walk(processed_data_dir):
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

    input_path = processed_data_dir + '/input/'
    labels_path = processed_data_dir + '/labels/'
    train_path = root_path + '/TRAIN/'
    val_path = root_path + '/VAL/'

    if not os.isdir(train_path):
        os.mkdir(train_path)

    if not os.isdir(val_path):
        os.mkdir(val_path)

    for seq in train_dirs_list:
        shutil.copytree(os.path.join(input_path,seq),os.path.join(train_path,seq))

    for seq in val_dirs_list:
        shutil.copytree(os.path.join(val_path,seq),os.path.join(val_path,seq))

train_val_fun(root_path=root_path)
import os
import random
import shutil

root_path = '/scratch/s174411/30_no_over'

def remove_duplicates(l):
    seen = {}
    res = []
    for item in l:
        if item in seen:
            res.append(item)
        else:
            seen[item] = 1
    return res

def train_val_fun(root_path, val_proportion = 0.1, test_proportion = 0.0):
    processed_data_dir = root_path + '/1D_MASS_MODA_processed'
    input_path = processed_data_dir + '/input/'
    labels_path = processed_data_dir + '/labels/'
    train_path = root_path + '/TRAIN/'
    val_path = root_path + '/VAL/'
    test_path = root_path + '/TEST/'
    train_input_path = root_path + '/TRAIN/input/'
    train_label_path = root_path + '/TRAIN/labels/'
    val_input_path = root_path + '/VAL/input/'
    val_label_path = root_path + '/VAL/labels/'
    test_input_path = root_path + '/TEST/input/'
    test_label_path = root_path + '/TEST/labels/'

    dirs_list = []
    ss_sizes = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            if dir[0] != '0':
                continue 
            dirs_list.append(dir)
            ss_sizes[dir[4]] += 1
    
    val_size = {}
    test_size = {}
    for k, v in ss_sizes.items():
        val_size[k] = int(v*val_proportion)
        test_size[k] = int(v*test_proportion)

    train_dirs_list = dirs_list.copy()
    val_dirs_list = []
    test_dirs_list = []
    val_creation = True
    while val_creation:
        seq_choice = random.choice(train_dirs_list)
        while val_size[seq_choice[-6]] == 0:
            seq_choice = random.choice(train_dirs_list)
        val_size[seq_choice[-6]] -= 1
        val_dirs_list.append(seq_choice)
        train_dirs_list.remove(seq_choice)
        loop_check = False
        for k,v in val_size.items():
            if v > 0:
                loop_check = True
        if not loop_check:
            val_creation = False
            
    if test_proportion > 0:
        test_creation = True
        while test_creation:
            seq_choice = random.choice(train_dirs_list)
            while test_size[seq_choice[-6]] == 0:
                seq_choice = random.choice(train_dirs_list)
            test_size[seq_choice[-6]] -= 1
            test_dirs_list.append(seq_choice)
            train_dirs_list.remove(seq_choice)
            loop_check = False
            for k,v in test_size.items():
                if v > 0:
                    loop_check = True
            if not loop_check:
                test_creation = False


    dupli = remove_duplicates(train_dirs_list)
    print(dupli)

    

    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    if not os.path.isdir(val_path):
        os.mkdir(val_path)

    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    if not os.path.isdir(train_input_path):
        os.mkdir(train_input_path)

    if not os.path.isdir(train_label_path):
        os.mkdir(train_label_path)

    if not os.path.isdir(val_input_path):
        os.mkdir(val_input_path)

    if not os.path.isdir(val_label_path):
        os.mkdir(val_label_path)

    if not os.path.isdir(test_input_path):
        os.mkdir(test_input_path)

    if not os.path.isdir(test_label_path):
        os.mkdir(test_label_path)



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

    if test_proportion > 0:
        for seq in test_dirs_list:
            dest_input = os.path.join(test_input_path,seq)
            dest_labels = os.path.join(test_label_path,seq)
            shutil.copytree(os.path.join(input_path,seq), dest_input)
            shutil.copytree(os.path.join(labels_path,seq), dest_labels)

train_val_fun(root_path=root_path)
import argparse
import os
import pickle
import pandas as pd
import itertools
from more_itertools import unique_everseen
import time

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-inpath',
                        required=True,
                        default='training_data/',
                        help='input file path.')

    return parser.parse_args()

def train_list_generate(files, feature_path, index, state):

    df_total = pd.read_csv(feature_path)
    df_tmp = df_total.loc[df_total[state] == index].reset_index(drop=True)
    # print(df_tmp)
    protein_list = df_tmp['protein'].tolist()
    # print(len(protein_list))
    protein_list = list(unique_everseen(protein_list))# remove the duplicated protein name in list but keep order
    print(len(protein_list))
    prot_list_tmp = []
    for prot in protein_list:
        prot_name = prot + '_'
        matching = [s for s in files if prot_name in s]
        # print(len(matching))
        if len(matching) != 0:
            df_prot = df_tmp.loc[df_tmp['protein'] == prot].reset_index(drop=True)
            # print(df_prot)
            drug_list_tmp = df_prot['drug'].tolist()
            list_tmp = []
            for i in range(len(drug_list_tmp)):
                name_tmp = prot_name + drug_list_tmp[i] + '_'
                matching_tmp = [s for s in matching if name_tmp in s]
                # print(matching_tmp)
                # print(len(matching_tmp))
                if len(matching_tmp) != 0:
                    list_tmp.append(matching_tmp)
                else:
                    print(matching_tmp, 'has no pair in drug protein pair')
            prot_list = list(set(itertools.chain.from_iterable(list_tmp)))
            # print(prot_name, 'has number of pair:', len(prot_list))
            # print(prot_list)
            prot_list_tmp.append(prot_list)
        else:
            print(prot_name, 'has no pairs in target')
    # print(prot_list_tmp)
    valid_list = list(set(itertools.chain.from_iterable(prot_list_tmp)))
    # print('length of valid list', len(valid_list), 'in cluster', index)
    # train_list = list(set(files) - set(valid_list))

    return valid_list

def save_traindataset_list(path):

    for dir in os.listdir(path):
        if 'positive' in dir:
            feature_path_p = path + dir
            files_p = os.listdir(feature_path_p)
            print(dir, 'number of files in dir:', len(files_p))

        if 'negative' in dir:
            feature_path_n = path + dir
            files_n = os.listdir(feature_path_n)
            print(dir, 'number of files in dir:', len(files_n))
    files_total = files_p + files_n
    print(files_total[0:10])
    print(len(files_total))
    inputfile = 'train_validation_list.csv'
    inx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for index in inx_list:
        cluster_valid_list = train_list_generate(files_total, inputfile, index, 'cluster')
        print(str(index) + '_valid has length:', len(cluster_valid_list))
        random_valid_list = train_list_generate(files_total, inputfile, index, 'random')
        print(str(index) + '_valid has length:', len(random_valid_list))

        cluster_path = 'input_list_cluster/'
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        random_path = 'input_list_random/'
        if not os.path.exists(random_path):
            os.makedirs(random_path)
        # print('train list length is ', len(train_list))
        # print('valid list length is ', len(valid_list))
        with open(cluster_path + str(index) + '_valid_list.pkl', 'wb') as f1:
            pickle.dump(cluster_valid_list, f1)
        with open(random_path + str(index) + '_valid_list.pkl', 'wb') as f2:
            pickle.dump(random_valid_list, f2)

if __name__ == "__main__":

    start = time.time()
    parse = getArgs()
    input_folder = parse.inpath
    save_traindataset_list(input_folder)
    end = time.time()
    print('vector time elapsed :' + str(end - start))

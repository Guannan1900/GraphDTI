"""
This code is used for predict test dataset for GraphDTI model. 
The input file should be the test list, the label list of the test dataset and the selected feature list index
The input test dataset should be the integration of Mol2vec features for the drugs, ProtVec, Bionoi-AE and Graph2vec features for the proteins

@author: lgn
"""
import os
import numpy as np
import torch
from torch.utils import data
from sklearn.metrics import average_precision_score,accuracy_score, roc_curve, auc
import pickle
import json
import argparse

from data_generator import Dataset

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir',
                        required=True,
                        default='test_data/',
                        help='input test data path')
    parser.add_argument('-model',
                        required=True,
                        default='GraphDTI.pt',
                        help='the pretrained models')
    parser.add_argument('-output',
                        required=False,
                        default='test_results/',
                        help='ROC-AUC results of test dataset')

    return parser.parse_args()

def to_onehot(yy):
    yy1 = np.zeros([len(yy), 2])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1

def class_probabilities_test(model, device, test_loader, class_num):

    model.eval()
    scores = np.empty((0, class_num))
    y_target = np.empty((0, class_num))
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum()
            # print(test_correct)
            out_result = outputs.cpu().numpy()
            out_target = target.cpu()
            prediction_prob = out_result[:, 1]
            prediction_binary = np.where(prediction_prob > 0.5, 1, 0)
            acc_mlp = accuracy_score(out_target, prediction_binary)
            # print(acc_mlp)
            scores = np.append(scores, out_result, axis=0)
            out_target = out_target.tolist()
            # print(out_target)
            labels_onehot = to_onehot(out_target)
            y_target = np.append(y_target, labels_onehot, axis=0)
        test_acc = test_correct.item() / test_total
        # print("Test Accuracy: %.4f " % (test_acc))

    return scores, y_target, test_acc


def mlp_test(data_path, model_path, output_path):

    batch_size = 1000
    n_classes = 2

    with open('permu_feature_importance.json') as json_file:
        feature_importance = json.load(json_file)
    feature_list = []
    score_list = []
    for name, improtance in feature_importance.items():
        feature_list.append(name)
        score_list.append(improtance)

    feature_tmp = feature_list[0:400]
    feature_sample = list(map(int, feature_tmp))
    # print(len(feature_sample))

    with open('test_list.pkl', 'rb') as f_test:
        new_test_list = pickle.load(f_test)
    print('The size of the test dataset', len(new_test_list))
    with open('test_label.pickle', 'rb') as f_label:
        labels = pickle.load(f_label)

    test_dataset = Dataset(new_test_list, labels, data_path, feature_sample)
    print(data_path)
    test_generator = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    mlp = torch.load(model_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # data_size = len(labels)
    our_scores, our_target, test_accuracy = class_probabilities_test(mlp, device, test_generator, n_classes)
    # print(our_scores.shape, our_target.shape)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresh = dict()
    for i in range(n_classes):
        y_score = np.array(our_scores[:, i])
        y_test = np.array(our_target[:, i])
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    precision = average_precision_score(y_test.ravel(), y_score.ravel())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.exists(output_path + 'roc_auc.pkl'):
        os.remove(output_path + 'roc_auc.pkl')
    if os.path.exists(output_path + "fpr.pkl"):
        os.remove(output_path + "fpr.pkl")
    if os.path.exists(output_path + "tpr.pkl"):
        os.remove(output_path + "tpr.pkl")
    if os.path.exists(output_path + "thresh.pkl"):
        os.remove(output_path + "thresh.pkl")
    # print(roc_auc)
    f1 = open(output_path + 'roc_auc.pkl', "wb")
    pickle.dump(roc_auc, f1)
    f1.close()
    f2 = open(output_path + "fpr.pkl", "wb")
    pickle.dump(fpr, f2)
    f2.close()
    f3 = open(output_path + "tpr.pkl", "wb")
    pickle.dump(tpr, f3)
    f3.close()
    f4 = open(output_path + "thresh.pkl", "wb")
    pickle.dump(thresh, f4)
    f4.close()

    return test_accuracy, roc_auc[1], precision

if __name__ == "__main__":

    parse = getArgs()
    model_path = parse.model
    print('Load the pretrained model', model_path)
    acc_mean, auc_mean, precision_mean = mlp_test(parse.data_dir, model_path,
                                                  parse.output)
    print('The AUC score of the test dataset is', auc_mean)


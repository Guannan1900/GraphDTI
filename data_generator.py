from torch.utils import data
import numpy as np


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, path, feature_list):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = path
        self.feature_list = feature_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        y = self.labels[ID]
        f_list = self.feature_list

        if y == 1:
            folder = self.path + 'positive/'
        elif y == 0:
            folder = self.path + 'negative/'

        # Load data and get label
        X = np.load(folder + ID, allow_pickle=True)
        X_feature = X[self.feature_list]
        # print(X_feature)
        # print(X_feature.shape)

        return X_feature, y
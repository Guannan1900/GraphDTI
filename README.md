# GraphDTI
## Description
GraphDTI is a deep learning framework to predict the drug-target interactions using the integrated four types of features, which contain the information of drugs, namely [Mol2vec](https://github.com/samoturk/mol2vec); the information proteins, namely [ProtVec](https://github.com/kyu999/biovec); the binding sites of proteins, namely [Bionoi-AE](https://github.com/CSBG-LSU/BionoiNet); and the information of the drug perturbed local protein-protein interaction(PPI) network of target proteins, namely [Graph2vec](https://github.com/benedekrozemberczki/graph2vec). Based on the local PPI network, we provided a method to optimize the graph2vec features. The details of feature generation and optimization can be found in https://github.com/Guannan1900/GraphDTI_preprocess. The deep neural network(DNN) is implemented based on [Pytorch](https://pytorch.org/). The pipeline of GraphDTI is shown below:

## Files
- ```permu_feature_importance.json```: The indices and important scores of all the integrated features. We provide a feature selection processdure in order to mitigate the overfitting problem. The details can be found in https://github.com/Guannan1900/GraphDTI_preprocess/tree/master/feature_selection.
- ```train_validation_list.csv```: The information of split protocols for cross-validation. We design two types of protocols for cross-validation in order to evaluate the generalizability of the model. The details can be found in https://github.com/Guannan1900/GraphDTI_preprocess/tree/master/clustering.
- ```training_label.pickle```: The labels for all instances in the training dataset. 
- ```test_list.pkl``` and ```test_label.pickle```: The names and labels for all instances in the test dataset.

> "Note: the training and test dataset can be available at: https://osf.io/ugvd9/"

## Usage
The framework is provided the training and prediction models
1. The training model
- Split the training and validation data of two protocols for cross-validation
  + required file: ```train_validation_list.csv```
  + input: the training data path, ```training_data/```
  + output: the validation name list in tow fold of two protocols for cross-validation, ```input_list_cluster/``` and ```input_list_random/```

```shell
python train_validation_list_generation.py -inpath 'training_data/'
```

- Train the multilayer perceptron(MLP) for cross validation
  + required files:
    * ```permu_feature_importance.json```
    * ```input_list_cluster/``` or ```input_list_random/```
    * ```training_label.pickle```
  + input:
    * -data_dir: the training data path
    * -bs: batch size, recommended value is 32
    * -lr: learning rate, recommended value is 0.0001
    * -epoch: number of epoch for taining, recommended value is 30
    * -protocol: two protocols for cross-validation, which are 'cluster' or 'random'
  + output:
    * -output: location for the trained models to be saved
    * -logs: location for the logs to be saved
```shell
python train.py -data_dir 'training_data/' -protocol 'cluster'
```

2. The prediction model
- Using pre-trained model for prediction on a test dataset
  + required files:
    * ```permu_feature_importance.json```
    * ```test_list.pkl```
    * ```test_label.pickle```
  + input:
    * -data_dir: a test dataset path
    * -model: the pre-trained model for GraphDTI
  + output:
    * -output: The location for the roc results on a test dataset

```shell
python predict.py -data_dir 'test_data/' -model 'GraphDTI.pt'
```
## Prerequisites
1. Python 3.7 or higher
2. Pytorch 1.2.0 or higher
3. Numpy 1.17.3
5. CUDA if you would like to run on GPU(s)

setwd("/nfsmb/koll/probst/r-packages/thundersvm/R/")
source("svm.R")
svm_train_R(dataset = "../dataset/test_dataset.txt", model_file = "../dataset/test_dataset.txt.model", cost = 100, gamma = 0.5)
svm_predict_R(test_dataset = "../dataset/test_dataset.txt", model_file = "../dataset/test_dataset.txt.model", out_file="test_dataset.txt.out")



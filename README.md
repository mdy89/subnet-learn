# subnet-learn
Subnetwork representation learning for discovering network biomarkers in predicting lymph node metastasis in early oral cancer

# Abstract
Cervical lymph node metastasis is a major factor in poor prognosis in patients with oral cancer, which occurs even in the early stages. The current clinical diagnosis depends on physical examinations that are not sufficient to determine whether micrometastasis remains. The transcriptome profiling technique has shown great potential for predicting micrometastasis by capturing the dynamic activation state of genes at the whole-genome scale. However, there are  technical challenges in using transcriptome data to predict the state of cancer tissue: 1) insufficient number of samples compared to the number of genes, 2) complex dependency between genes that control cancer phenotypes, and 3) heterogeneity among patients within geographically and ethnically different cohorts. We developed a  graph embedding and attention network method to overcome such challenges, which aims to extract subnetwork representation from cancer transcriptome and create a sophisticated prediction model based on the learned representation. The method achieved high accuracy in predicting metastatic potential in oral cancer and succeeded to produce network biomarkers having significant clinical implications. Remarkably, the network biomarkers for metastatic potential were performed well for two geographically and ethnically distant patient groups, TCGA and SNUH samples of Korean patients.

# Install
pip install -r requirements.txt

# Usage
1) Identification of Subnetworks: Subnetworks are extracted from a given set of genes and given PPI information.

example code: python3 subnetwork_identification.py test_geneset_list.txt test_ppi_list.txt test.out

** DeepWalk must be installed and can be called from bash: https://github.com/phanein/deepwalk

1) Calculating sSAS: fitting sSAS model with train data then transforming train and test data with the model

python3 sSAS_fit_transform.py [SUBNETWORK_LIST.txt] [TRAIN_TPM_LIST.txt] [TEST_TPM_LIST.txt] [sSAS_OUTPUT.dir]

(see test.sh)

2) Constructing SRL: fitting SRL model with train data then transforming train and test data with the model with generating classification report

python3 SRL_fit_transform.py [sSAS_OUTPUT.dir] [SRL_OUTPUT.dir]

(see test.sh)

# Result files
1) subnetwork_evaluated.txt: feature evaluation results during sSAS calculation process (it's different than the SRL attention weights)

2) classification_report.txt: summary of performance of the trained SRL in terms of classification power

3) train_set_attention_map.txt: list of attention weights of each subnetwork in the trained SRL when passing the train data

4) train_set_predictions.txt: prediction results of the trained SRL when passing the train data

5) test_set_attention_map.txt: list of attention weights of each subnetwork in the trained SRL when passing the test data

6) test_set_predictions.txt: prediction results of the trained SRL when passing the test data




# Contact
mindoly89@gmail.com

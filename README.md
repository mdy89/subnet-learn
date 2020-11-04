# subnet-learn
Subnetwork representation learning for discovering network biomarkers in predicting lymph node metastasis in early oral cancer

# Abstract
Cervical lymph node metastasis is a major factor in poor prognosis in patients with oral cancer, which occurs even in the early stages. The current clinical diagnosis depends on physical examinations that are not sufficient to determine whether micrometastasis remains. The transcriptome profiling technique has shown great potential for predicting micrometastasis by capturing the dynamic activation state of genes at the whole-genome scale. However, there are  technical challenges in using transcriptome data to predict the state of cancer tissue: 1) insufficient number of samples compared to the number of genes, 2) complex dependency between genes that control cancer phenotypes, and 3) heterogeneity among patients within geographically and ethnically different cohorts. We developed a  graph embedding and attention network method to overcome such challenges, which aims to extract subnetwork representation from cancer transcriptome and create a sophisticated prediction model based on the learned representation. The method achieved high accuracy in predicting metastatic potential in oral cancer and succeeded to produce network biomarkers having significant clinical implications. Remarkably, the network biomarkers for metastatic potential were performed well for two geographically and ethnically distant patient groups, TCGA and SNUH samples of Korean patients.

# Usage
1) Constructing subnetwork representations for RNA-seq data
python construct_subnetwork_representation.py [GENESET_INFORMATION.txt] [NORMALIZED_GENE_EXPRESSION_TABLE.txt] [OUTPUT_DIR]

2) Constructing attention networks based predictor
python construct_predictor.py --input_table [SUBNETWORK_REPRESENTATION_TABLE.txt] --cross_validation [NUMBER] --out_dir [OUTPUT_DIR] --process_number [NUMBER]

# Contact
mindoly89@gmail.com

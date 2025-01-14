--------------------
DATA & FILE OVERVIEW
--------------------
MATLAB codes and data used to evaluate cellular and tridimensional electro-mechanical simulations of 39 different compounds.

DATA:
	BIOMT_CiPA_EM_0D.mat --> Biomarker results for cellular EM simulations of 28 CiPA compounds plus control conditions.

	BIOMT_CiPA_EM_3D.mat --> Biomarker results for tridimensional EM simulations of 28 CiPA compounds plus control conditions.

FOLDERS:
	
	#1 - CCBs_biom --> Folder with one matlab data file for tridimensional EM biomarker results for each of 11 CCBs compounds tested.	
	#2 - models --> Folder with each one of prediction models showed in the paper. Also includes the codes used to preprocess input data and obtain validation prediction of each model. 

	Codes:
		predictionDrugs.m --> Code to obtain the prediction of validation CiPA drugs for any prediction model included. Selecting "val_BIOMT","SVM_classifier" and  "LogisticRegressionClassifiers" included in the "models" folder.
		preProcessSimData.m --> Code use to preprocess biomarker results (BIOMT_CiPA_EM_0D.mat and BIOMT_CiPA_EM_3D.mat) to build classification models.
	
	


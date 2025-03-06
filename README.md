# Repository for <b>SAMMY</b>, a CNN Based Model to Predict Unlabeled Mammography Metadata (Left/Right, MLO/CC)
### <b>SAMMY</b> (Small Automated Mammography Metadata Yielder)

### Utils Folder
#### data_tools.py
Contains cbis_ddsm and inbreast classes for loading and handling the datasets
#### featurizer.py
Contains the functions for featurizing mammographys for feature based prediction
#### models.py
Contains class to test models on data (ResNet50, RandomForest, SmallCNN <b>(SAMMY)</b>)
#### Settings.py
Contains settings for image compression and dataset path management

### Tests Folder
#### random_forest.py
Runs a grid search of random forests on the cbis_ddsm dataset and validates the top models with inbreast, top model 87% on inbreast
#### small_cnn.py <b>(SAMMY)</b>
Trains a cnn on the cbis_ddsm dataset and validates on inbreast, 99% accuracy on inbreast

Datasets Used:
Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

Moreira IC, Amaral I, Domingues I, Cardoso A, Cardoso MJ, Cardoso JS. INbreast: toward a full-field digital mammographic database. Acad Radiol. 2012 Feb;19(2):236-48. doi: 10.1016/j.acra.2011.09.014. Epub 2011 Nov 10. PMID: 22078258.

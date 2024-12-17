# Repository for <b>SAMMY</b>, a CNN Based Model to Predict Unlabeled Mammography Metadata (Left/Right, MLO/CC)
### <b>SAMMY</b> (Smart Automated Mammography Metadata Yielder)

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

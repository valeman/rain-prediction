# Bangladesh Rain Prediction (CatBoost, Conformal Prediction, F1 Score:0.84)

original data -> http://data.gov.bd/dataset/live-weather-condition

It is an overt fact that the <b>Rainfall</b> feature is very highly correlated with the target variable in respect of Mutual Information score and Point-BiSerial Coefficient. This situation can give rise to an overfitting problem. For example, I used  the Rainfall feature in training, model metrics quickly increased and to be 100 % in my trial with a CatBoostModel. This problem in machine learning is the target leakage [1,2]. This is why I dropped the <b>Rainfall</b> feature.

I have used the following methods.

* Feature engineering for timeseries data (creating <b>Cyclical Features</b>),
* Feature selection for solving <b>the target leakage problem</b> (dropping Rainfall feature),
* A tuned CatBoost Model (with optuna),
* Feature Explanation with <b>SHAP</b>,
* <b>Conformal Prediction</b> for calibration,


## My Another Projects
* [(76 GB) 160 Polish Bird Sounds Classification](https://www.kaggle.com/code/banddaniel/76-gb-160-polish-bird-sounds-classification)
* [Segment Medical Instrument, w/Custom DeepLabv3+(Dice: 0.86)](https://www.kaggle.com/code/banddaniel/segment-medical-instrument-deeplabv3-dice-0-86)
* [Rice Classification w/Custom ResNet50 (ACC 85%)](https://www.kaggle.com/code/banddaniel/rice-classification-w-custom-resnet50-acc-85)


## References
1. https://en.wikipedia.org/wiki/Leakage_(machine_learning)
2. https://dataintegration.info/detect-multicollinearity-target-leakage-and-feature-correlation-with-amazon-sagemaker-data-wrangler
3. https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html

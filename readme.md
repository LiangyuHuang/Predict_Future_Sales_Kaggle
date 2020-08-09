# Experimentation


 This program needs variety of libraries.
 
 These libraries would included numpy, pandas, sklearn, math, zipfile, urllib3, xgboost, lightgbm, catboost, os, itertools.
 
 This program may need 50 minutes to run, but uncomment the line 164 'task_type='GPU'' if you have GPU
 
 The needed dataset could download from the kaggle website automatically.  If any change in download link causes an error, please go to the web page to download and unzip all CSV files into the py directory. 
 website: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

 If you need to check the RMSE of all models and stacking, please follow the following tips:
 1) comment the line 72-100  #used for parameters selection
 2) comment the line 172-174  #fitting and check the R_MSE of each model
 3) comment the line 176-184  #test stacking performance

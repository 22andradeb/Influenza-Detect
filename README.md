# Influenza-Detect
# Worldwide Influenza Outbreak Detection


### Project Structure:
- Clinical Problem/Storyline
- Initial Assumptions
- Dataset Characteristics
- Experementation
- Mid-Project planning
- Preprocessing
- Baseline Models
- Improvements
- Final Metrics
- Conclusion

##Install

Prereqs: conda/mamba, Git, and a working Jupyter setup ( can work in pycharm or any enviornment).
Create the environment (recommended via conda/mamba):
conda env create -f environment.yml
conda activate 
Verify Jupyter + widgets:
python -c "import IPython, ipywidgets; print('Jupyter OK')"
If using JupyterLab and widgets don’t render, ensure Lab ≥ 3.x. No manual widget install is typically required with this env.



Health Organizations around the global struggle to properly detect illness within their clients, particularly those with influenza. Their issue is understanding when to expect an influenza outbreak. Countries all over the world have not been able to decide on how to accurately detect influenza outbreaks. Our goal is to provide the world health organization as well as other health centers, insights on what surveillance type is more effective and where to allocate resources



### Caveats (Things that can impact the analysis):
There could be missing values, N/A data which is common in datasets obtained from surveillance
Some countries have better surveillance infrastructure than others which can affect how data is obtained which may impact any future results
Lack of individual level data can result in relevant factors being ignored from the models.
Certain unique values or variables could weight more than other which introduces biases in the analysis.

EDA (Exploratory Data Analysis):

10,835 values and 16 columns (8 once we drop the columns with Null values)


Variables:

Country area or territory: Country reporting influenza
Surveillance site type: Sentinel vs Non-sentinel site
Year-week/Week start date: Temporal identifier (ISO 8601 week)
Specimen tested: Number of samples tested
Influenza positive / Influenza negative: Counts of Positive vs Negative test results.
Subtypes Counts (these variables are the ones containing missing values.


By generating a histogram of the missing values of the variables, we see that the dataset only has empty values for the subtypes of Influenza. Given that the subtype variables are not necessary for our clinical story, we decided to drop the variables for our new clean dataset.


After reviewing the unique values of the categorical variables, we find that there are several values. We find that ‘Surveillance site type’ has three unique values: ‘Sentinel’, ‘Non-sentinel’, and ‘Not defined’. The Country variable has far too many unique values to analyze.


Once we summarize the statistics of the numerical variables of ‘Influenza Positive’ and ‘Influenza Negative’, we learn that there are a lot more negative cases of tested individuals in the dataset than there are positive cases.


We can see after comparing the numerical and categorical variables that despite the fact that there are more sentinel values than there are non-sentinel values, most of the tested values with either positive or negative values come from non-sentinel surveillance.


### Next Steps:

Because our dataset has surveillance level data and not patient level data (i.e. Age, sex, symptoms, outcomes, vaccination status, etc.) we will focus on basing our clinical story on a population or system rather than an individual diagnosis.


Our possible analytical directions for how we craft our story is through predictive surveillance efficiency (predicting which surveillance type is more effective at detecting influenza), evaluate surveillance sensitivity (which surveillance type detects higher proportion of positive cases using regression analysis), and forecasting influenza positivity (forecast influenza positivity for future weeks using past data).


We can base our focus for our clinic story on epidemiology (how do surveillance strategies differ in influenza detection efficacy), predictive modeling (How can we anticipate influenza surges based on surveillance and testing patterns), and operation managment (which testing strategy yields the highest return and where should resources be prioritized)



### Preprocessing steps:

Before training our models we need to perform data preprocessing so that we can use it efficiently and without unnecessary factors affecting our results. Preprocessing is always a necessary step when it comes to building a reliable machine learning pipeline.


First we need to clean the dataset of any missing values. N/A or null values can create noise, making it harder for the model to detect accurate high-level influenza trends. After performing our EDA we discovered that only the columns for the various subtypes of influenza contained the missing values. Because we wished to retain the large amount of data while eliminating the missing values, we dropped the subtype columns for effective machine learning analysis.


Second we kept the core variables needed for our models within specific defined variables of code, a technique known as feature selection. The key predictors we kept were ‘Surveillance_site_type’, ‘Influenza_positive’, ‘Influenza_negative’, and ‘Specimens_tested’. These variables describe surveillance efficiency and disease detection which is vital to our analysis goal.


Next we performed one-hot encoding in order to convert the categorical data such as the sentinel and non-sentinel values into numerical form. Models can only process numerical input so encoding categorical variables allows us to include values such as the surveillance site types as a predictive factor.


Afterwards we applied a StandardScaler into our pipeline in order to scale the numerical features. What this does is ensure that all features have similar scales which prevents variables with large ranges (such as counts of specimen tested) from dominating the model.


Lastly we divide the data into training, testing and validation sets (70%, 15%, 15%). This simple technique prevents overfitting by ensuring the model is evaluated on unseen data.


### Our Initial Models:

Our regression model is designed to estimate the positivity rate or number of positive influenza cases based on surveillance and specimen data. It does this by learning a mathematical relationship between predictors (such as specimen tested and surveillance type) and a continuous outcome (like positivity rate). Our simplest form of regression is linear regression, which fits a line through the data minimizing squared error. This model is useful for understanding which surveillance or testing conditions predict higher influenza activity.


The function for the classification model is to classify whether a specimen will test positive for influenza based on site type and testing data. To do this the model learns to separate cases into two classes (positives and negatives), uses logistic regression to produce probabilities that can be thresholded for predicting a positive. It’s insightful that the model can identify whether sentinel sites or testing volume affect influenza detection success.


Lastly for the forecasting model, we forecast future positivity rates to anticipate upcoming influenza waves. The model uses temporal relationships (time ordered data) in a lag-based linear regression where next week’s positivity depends on last week’s rate. An increasing slope would mean an outbreak is surging, a decreasing slope would mean influenza transmission is declining. The model can give public health officials an early warning, predicting peaks and helping allocate resources.



### Metrics:


For the regression model (in which we attempt to predict the influenza positivity rate), we use R² (Coefficient of Determination), MAE (Mean Absolute Error), and MSE (Mean Squared Error). R² measures how much variance in the target is explained by the model, MAE is the average absolute difference between predicted and actual values, and MSE measures the average prediction error. For example: our initial influenza positivity model obtained R² = 0.3, MAE = 0.1 and MSE = 0.02. This isn’t good even as a start as the R² is too low and the MAE is a tad high, the MSE is low which is fine. We can see that the model will need to be improved.


In the case of the classification model (which predicts categories such as whether a specimen will test positive or negative and whether a sentinel site detects cases earlier), we use Accuracy, Precision, Recall, and ROC-AUC (Area under the Receiver Operating Characteristic Curve). Accuracy is the percentage of correct predictions, Precision is of all predicted positives, how many were actually positives, Recall: of all actual positives, how many did we catch, and ROC_AUC is the area under the curve of true positive rate vs false positive rate. For example, our initial sentinel-site classifier model obtained an Accuracy = 0.68, F1 Score = 0.67 and ROC_AUC = 0.74. This isn’t a good start as the Accuracy and F1 Score are mediocre at best, and the ROC_AUC is only decent. This model will need to go through an improvement.


After hyperparameter tuning with Random Forest, the regression model achieved a score of R² = 0.97, and RMSE = 0.02. This means that the model can accurately estimate influenza trends based on surveillance data. So what? Our model can be used as a predictive surveillance tool to forecast where influenza cases are likely to increase.


For the forecasting model which predicts influenza outbreaks, the model, using random forest regression, was able to obtain a R² = 0.99 and RMSE = 0.006. After running a Influenza surge function with the model implemented, we were able to predict for the next 8 weeks no imminent outbreak, the expected positivity remains low and stable.


With a new classification model using a random forest classifier and establishing a positivity rate threshold of positivity rate > 0.10 so that the model would understand how much influenza positive cases would be considered an outbreak, the model obtained an Accuracy = 0.96, an Precision = 0.93, Recall = 0.98, and an ROC AUC = 0.96. What this means is that the model can accurately predict whether a given week will experience an influenza outbreak.


Story Line: Our team will analyze the data collected by the World Health Organization from various countries and prepare a machine learning model that can decide on which type of surveillance site can best detect influenza in a specimen so that health organisations are informed on where to allocate their resources.


### Conclusion:

With our two models, forecasting and classification, we are capable of anticipating how severe influenza circulation will be and interpret those results to trigger public health alerts for potential outbreaks. This joint system allows organizations such as WHO, CDC and others to detect patterns of increasing influenza positivity in real time and prepare vaccine distribution, hospital staffing, trigger prevention alerts in high risk areas and use for evaluation of certain situations.



Source: World Health Organization - https://worldhealthorg.shinyapps.io/flunetchart/ 



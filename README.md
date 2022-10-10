# Predictive-Analysis-on-Startup-Acquisition-Status
This project predicts a startup’s acquisition status based on its financial statistics. In order to overcome
the main challenge of biased data without under/oversampling.


## Understanding the Dataset 
The dataset used for this project is sourced from Crunchbase called ‘Crunchbase 2013 - Companies,
Investors, etc.

 There are n = 196553 samples and each row of the dataset contains a startup’s information.
Specifically, these are: company name, website, sector category, funding received, headquarter location (city and
state names), funding rounds, founding date, first and last funding dates, and last milestone date. Each row is also
labeled with the company’s status (‘Acquired’, ‘Closed’, ‘IPO’, ‘Operating’). 

The dataset labels show that the dataset is extremely biased. The ‘Operating’ class is extremely over-represented and the other classes are
under-represented


## Preprocessing 
There are 44 columns in the dataset. 

Remove the columns such as id, Unnamed: 0.1, entity type, entity id, parent id, created by, created-at, updated-at, domain, homepage url, Twitter username, logo url, logo width, logo-height, short description, description, overview, tag list, name, normalized name, permalink, and invested companies because they contain irrelevant information.

Filtered the datset by deleting the duplicate values.

Columns like first investment at, last investment at, investment rounds, and ROI, which include more than 96% of null values, were checked for null values and eliminated.

Delete instances with missing values for 'status', 'country_code', 'category_code' and 'founded_at'.Since these are the type of data where adding value via imputation will create wrong pattern only.

Checked for outliers and deleted outliers for 'funding_total_usd' and 'funding_rounds'.

Coverted qualitative data to quantitative data as part of feature extraction.

The derived columns are activedays and isclosed.


## EDA and Feature Engineering

**Introduction:**

- **merged_data** dataset comprises of 196553 rows and 44 columns.
- Dataset comprises of continious variable and float data type. 

**Information of Dataset:**
Using scatterplot founded that there is no correlation between funding_total_usd and relationships and also between milestones and relationships.

Using barplot between status and funding_total_usd, it is clear that funding_total_usd is higher for IPO status.

Using barplot between status and milestones, it is clear that milestones is higher for IPO status.

Using countplot on target variable **Status** we could see that Label 0 has '453' values , Label 1 has '6000',  Label 2 has 90, and Label 3 has 936. By this information we could conclude that there is  imbalanced in the data and hence balancing of data is required.

Using countplot on target variable **isclosed** we could see that Label 0 has '1389' values , Label 1 has '6090'. By this information we could conclude that there is imbalanced in the data and hence balancing of data is required.

Generalised the country and state columns and performed one hot encoding for country and state columns.

**Univariate Analysis:**

By plotting distplot it is evident that funding_total_usd, activedays are right skewed.

**Descriptive Statistics:**

Using **describe()** we could get the following result for the numerical features

        funding_rounds	funding_total_usd	milestones	relationships	lat	            lng
count	22889.000000	2.046700e+04	35249.00000	    48306.000000	61219.000000	61219.000000
mean	1.805758	    1.582132e+07	1.41587	        4.442926	    37.293151       -50.708830
std	    1.310805	    6.990693e+07	0.73856	        13.266474	    15.812771	    70.783600
min	    1.000000	    2.910000e+02	1.00000	        1.000000	    -50.942326	    -159.485278
25%	    1.000000	    5.110380e+05	1.00000	        1.000000	    34.052234	    -112.028750
50%	    .000000	        2.725875e+06	1.00000	        2.000000	    39.739236	    -75.898684
75%	    2.000000	    1.200000e+07	2.00000	        4.000000	    45.417979	    1.801799
max	    15.000000	    5.700000e+09	9.00000	        1189.000000	    77.553604	    176.165130

Created a cluster with lat and lng columns but there is no signifcance in the mutual information score hence removed these columns.

**Correlation Plot of Numerical Variables:**

All the continuous independent variables are not much correlated with each other hence there is no multicollinearity in the dataset.


Before modelling and after splitting we scaled the data using standardization to shift the distribution to have a mean of zero and a standard deviation of one.
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_valid)
```
**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, but we want our test data to be a completely new and a surprise set for our model.


### PCA transformation
We reduced the 5 features to be only 4.

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca.fit(X_train)
trained = pca.transform(X_train)
transformed = pca.transform(X_train)

## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall


#### Random Forest Classifier
- The random forest is a classification algorithm consisting of **many decision trees.** It uses bagging and features randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
- **Bagging and Boosting**: In this method of merging the same type of predictions. Boosting is a method of merging different types of predictions. Bagging decreases variance, not bias, and solves over-fitting issues in a model. Boosting decreases bias, not variance.
- **Feature Randomness**:  In a normal decision tree, when it is time to split a node, we consider every possible feature and pick the one that produces the most separation between the observations in the left node vs. those in the right node. In contrast, each tree in a random forest can pick only from a random subset of features. This forces even more variation amongst the trees in the model and ultimately results in lower correlation across trees and more diversification.

#### Quadratic Discriminant Analysis
- QDA is a variant of LDA in which an individual covariance matrix is estimated for every class of observations. 
- QDA is particularly useful if there is prior knowledge that individual classes exhibit distinct covariances.
- In QDA, we need to estimate Σk for each class k∈{1,…,K} rather than assuming Σk=Σ as in LDA. The discriminant function of LDA is quadratic in x:
δk(x)=−12log|Σk|−12(x−μk)TΣ−1k(x−μk)+logπk.

In this analysis , there is two dependent varaibles('status' and 'isclosed'). 
**QDA** is used where **isclosed** is taken as a dependent variable and **Random forest** uses **status** as a dependent variable.

- When we apply **random forest** model the accuracy is 84% and when we apply **Quadratic discriminate analysis** the accuracy is 85%.

Created a pipeline object by providing with the list of steps. Our steps are — standard scalar and Quadratic Discriminant Analysis.
 These steps are list of tuples consisting of name and an instance of the transformer or estimator.
  Let’s see the piece of code below for clarification -

pipeline_qda=Pipeline([('std', StandardScaler()),('pca', PCA(n_components = 4)),('qda_classifier',QuadraticDiscriminantAnalysis(reg_param=0.01, store_covariance=True))]

Also created another pipeline object by providing with the list of steps. Our steps are — standard scalar and Quadratic Discriminant Random Forest Classifier.
 Let’s see the piece of code below for clarification -

pipeline_randomforest=Pipeline([('std', StandardScaler()),('pca', PCA(n_components = 4)),('rf_classifier',RandomForestClassifier())])

After pipline creation , this model is saved in a picke file.

## Deployment
Access app by clicking the following link
https://startup-analysis.herokuapp.com/

### Flask 
Created app by using flask , then deployed it to Heroku . The files of this part are located into (Flask_deployment) folder. Access the app by following this link : [startup-analysis-flask](https://startup-analysis.herokuapp.com/)

### Heroku
Deploy the flask app to [ Heroku.com](https://www.heroku.com/). In this way, share app on the internet with others. 
Prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (app.py)  successfully 
- app.py: contains the python code of a flask web app.
- start_qda.pkl: contains our QDA model that built by modeling part.
- start_rf.pkl: contains our QDA model that built by modeling part.


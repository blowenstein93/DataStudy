#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from sklearn import linear_model
import seaborn as sb


# In[2]:


path = "./loan.csv"
inputDf = pd.read_csv(path, sep = ",", low_memory=False)


# In[3]:


selectedCols = inputDf.copy(deep = True)


# # Part One - Data Exploration

# ### parse data into numeric where relevant

# In[4]:


termDummies = pd.get_dummies(selectedCols["term"]).drop(columns = [" 36 months"])
selectedCols = termDummies.join(selectedCols.drop(columns = ["term"]))
selectedCols.head()


# In[5]:


selectedCols["origYear"] = selectedCols["issue_d"].str.extract("\w{3}-(\d{4})")
selectedCols = selectedCols.drop(columns= ["issue_d"])


# In[6]:


typedDf = selectedCols.copy(deep = True)


# ## look at distributions

# In[7]:


# looking at data types
typedDf.dtypes


# In[8]:


sb.catplot(x="grade", y="int_rate", data=typedDf, kind="box", order = ("A", "B", "C", "D", "E", "F","G"))
## worried that grades and interest rates lead to multicolinearity problems
## also notice that all grades except A and G have some sort of 'outlier' interest rate at 5%.


# In[9]:


# convert grades to numerics
gradeDummies = pd.get_dummies(selectedCols["grade"], prefix="grade")
selectedColsNoGrade = typedDf.drop(columns=["grade"])
dummiedGradesDf = gradeDummies.join(selectedColsNoGrade)
del selectedColsNoGrade
del selectedCols


# In[10]:


dummiedGradesDf["grade"] = dummiedGradesDf["grade_A"] * 0 +     dummiedGradesDf["grade_B"] * 2 + dummiedGradesDf["grade_C"] * 3 + dummiedGradesDf["grade_D"] * 4  +     dummiedGradesDf["grade_E"] * 5 + dummiedGradesDf["grade_F"] * 6 + dummiedGradesDf["grade_G"] * 6
dummiedGradesDf = dummiedGradesDf.drop(columns=["grade_A", "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G"])


# In[11]:


typedDf = dummiedGradesDf


# In[12]:


import matplotlib.pyplot as plt
sb.jointplot(x="funded_amnt", y="int_rate", data = typedDf, kind="hex", height =8)


# #### not surprising that annual income is highly skewed... lets reduce skew

# In[13]:


boxplot = typedDf.boxplot(column=['annual_inc'])


# In[14]:


incomeFilter = typedDf[typedDf["annual_inc"] < 300000]
sb.distplot(incomeFilter["annual_inc"].dropna())


# In[15]:


import matplotlib.pyplot as plt
sb.jointplot(x="funded_amnt", y="annual_inc", data = incomeFilter, kind="hex", height = 8)


# ### Look at Correlations

# In[16]:


sb.heatmap(typedDf[["loan_amnt", "funded_amnt",
                    "int_rate", "grade", "annual_inc", "dti",
                    "revol_bal", "total_pymnt", "loan_status"]].corr())


# funded amt and loan amount are highly correlated, this is trivial
# 
# interest rate and grade highly correlated (as seen above in box/whisker)

# ### look at distributions

# In[17]:


### everything is skewed right, except grade which is perfectly balanced
### DTI seems to have an outlier, or is 9999 a null placeholder ?
normalized = typedDf.copy(deep = True)
normalized.describe()


# In[18]:


#looks like dti= 9999 is a null placeholder because there are few values between ~ 1000 and that 9999 value
dti = typedDf[["dti"]]
boxplot = dti.boxplot(column=['dti'])


# In[19]:


#looks much better if we remove 9999 - still heavily skewed though.
dti[dti["dti"] < 900 ].boxplot(column=['dti'])


# In[20]:


# much nicer if we remove outliers ( > 300)
dti[dti["dti"] < 300 ].boxplot(column=['dti'])


# In[21]:


def normalize(df, col):
    ser = df[[col]]
    normalized_ser=(ser-ser.min())/(ser.max()-ser.min())
    df.drop(columns=[col])
    df[col] = normalized_ser
    return df


# In[22]:


normalizeDf = typedDf.copy(deep = True)

normalize(normalizeDf, "loan_amnt")
normalize(normalizeDf, "funded_amnt")
normalize(normalizeDf, "int_rate")
normalize(normalizeDf, "annual_inc")
normalize(normalizeDf, "dti")
normalize(normalizeDf, "revol_bal")
normalize(normalizeDf, "total_pymnt")
normalize(normalizeDf, "grade")
normalizeDf.boxplot(column=["loan_amnt", "funded_amnt", "int_rate", "annual_inc",
                                  "dti", "revol_bal", "total_pymnt", "grade"], figsize=(13,6))


# In[23]:


del normalized
del normalizeDf


# In[24]:


#Confirms suspicions of bad skew.
# annual income being the worst offender. nothing to do now other than just keep note for later stages
typedDf[["loan_amnt", "funded_amnt", "int_rate", "annual_inc",
          "dti", "revol_bal", "total_pymnt", "grade"]].skew(axis = 0)


# In[25]:


hist = typedDf[" 60 months"].hist(bins=2)
## most loans are 36 month


# In[26]:


revolveStudy = typedDf.copy(deep = True)
revolveStudy["revol_bal"] = pd.cut(typedDf["revol_bal"], bins = [-2, 10000, 25000, 5000000], labels = ["low", "medium", "high"])
sb.catplot(x="revol_bal", kind="count", data=revolveStudy)


# In[27]:


sb.catplot(x="home_ownership", kind="count", data=typedDf)


# # Part Two - Business Analysis
# #### this did not say to balance weight, but this should be balance weighted

# In[28]:


##only look at 36 month loans
shortTerm = typedDf[typedDf[" 60 months"] == 0]


# In[29]:


shortTerm.loan_status.value_counts()


# In[30]:


## only look at loans that are no longer alive 
# assuming that if they are done paying they are in one of three categories below
# not sure what: "Does not meet the credit policy. Status:Charged Off", "Does not meet the credit policy. Status:Charged Off"
shortTerm = shortTerm[(shortTerm["loan_status"] == "Fully Paid") | (shortTerm["loan_status"] == "Charged Off") | 
                    (shortTerm["loan_status"] == "Default")]


# In[31]:


shortTerm.head()


# #### Pct loans fully paid ?

# In[32]:


relFreqs = shortTerm["loan_status"].value_counts() / shortTerm["loan_status"].dropna().size
counts = shortTerm["loan_status"].value_counts()
print("there are {} loans fully paid, which represents {} loans".format(counts[0].round(4), relFreqs[0].round(4)))


# In[33]:


resultDf = pd.get_dummies(shortTerm["loan_status"])


# In[34]:


weightedPaidOff = (resultDf[["Fully Paid"]].astype(float).values *  shortTerm[["loan_amnt"]].astype(float).values).sum()


# In[35]:


wavgLoanPaidOff = weightedPaidOff / shortTerm[["loan_amnt"]].astype(float).values.sum()


# In[36]:


print("""The balance weighted avg of loans that paid off is {}, 
as this is slightly larger than the non-weighted pct of loans, we conclude the 
loans that pay off fully are higher balance than average""".format(wavgLoanPaidOff.round(4)))


# #### highest rates of default ? 

# In[37]:


defaultDf = shortTerm[["origYear", "grade", "loan_status", "loan_amnt"]]


# In[38]:


statusDummies = pd.get_dummies(defaultDf["loan_status"])
defaultDf["defaulted"] = (statusDummies[["Fully Paid"]] * - 1) + 1


# In[39]:


defaultGroupings = defaultDf.groupby(['origYear', 'grade']).mean()
cohorts = defaultGroupings.sort_values(by = "defaulted", ascending = False)
#defaultGroupings.unstack()
maxDefaults = cohorts.iloc[:1,]
print(maxDefaults)
print("the highest default rate of 0.48 was found among 2008 G's !")


# In[40]:


defaultDf["weightedAmnt"] = defaultDf["defaulted"].astype(float).values *  defaultDf["loan_amnt"].astype(float).values 
defaultWeighted = defaultDf.groupby(["origYear", "grade"]).agg({'weightedAmnt': ['sum'], 'loan_amnt': 'sum'})
defaultWeighted = defaultWeighted["weightedAmnt"].astype(float) / defaultWeighted["loan_amnt"].astype(float)
defaultWeighted.sort_values(by = "sum", ascending = False).head()


# ##### 2008 G's are still the highest default prob when weighted by loan size at 50% chance of default

# ### annualized rate of return

# In[41]:


rateReturn = shortTerm


# In[42]:


rateReturn["annualizedRateReturn"] = np.power((rateReturn["total_pymnt"] / rateReturn["funded_amnt"]), 1/3) - 1


# In[43]:


rateReturn = rateReturn[["annualizedRateReturn", "origYear", "grade"]]


# In[44]:


rateReturn.head()


# In[45]:


returnCohorts = rateReturn.groupby(['origYear', 'grade']).mean()
returnCohorts = returnCohorts.sort_values(by = "annualizedRateReturn", ascending = False)


# #### below is a summary of the annualized rate of return for each cohort (grouped by origination year and grade)

# In[46]:


rateReturn.groupby(['origYear', 'grade']).mean().unstack()


# # Part 3 - Modeling

# In[47]:


modelDf = typedDf.copy(deep = True)
modelDf.head()


# let's get rid of outliers we detected in part 1

# In[48]:


modelDf = modelDf[modelDf["dti"] < 300]


# In[49]:


modelDf = modelDf[modelDf["annual_inc"] < 300000]


# In[50]:


balCuts = pd.cut(modelDf["revol_bal"], bins = [-2, 10000, 25000, 5000000], labels = ["low", "medium", "high"])
modelDf = modelDf.join(pd.get_dummies(balCuts, prefix="revol_bal")).drop(columns = ["revol_bal"])


# In[51]:


modelDf = modelDf[(modelDf["loan_status"] == "Fully Paid") |
                                          (modelDf["loan_status"] == "Charged Off") |
                                          (modelDf["loan_status"] == "Default")]
statusDummies = pd.get_dummies(modelDf["loan_status"])
modelDf["defaulted"] = (statusDummies[["Fully Paid"]] * - 1) + 1


# ### first pass using the variables used in data analysis thus far

# In[52]:


firstPassRegression = modelDf[["loan_amnt", " 60 months", "int_rate", "annual_inc", 
         "dti", "revol_bal_low", "revol_bal_medium", "revol_bal_high", "total_pymnt", "loan_status"]]


# ### create features / labels

# In[53]:


firstPassRegression = firstPassRegression[(firstPassRegression["loan_status"] == "Fully Paid") |
                                          (firstPassRegression["loan_status"] == "Charged Off") |
                                          (firstPassRegression["loan_status"] == "Default")]
statusDummies = pd.get_dummies(firstPassRegression["loan_status"])
firstPassRegression["defaulted"] = (statusDummies[["Fully Paid"]] * - 1) + 1


# In[54]:


labels = firstPassRegression["defaulted"]
features = firstPassRegression.drop(columns=["defaulted", "loan_status"])
features.skew()


# ### split train and test

# In[55]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[56]:


train_features.describe()


# In[57]:


sb.heatmap(train_features.corr())


# ### Establish Baseline

# In[58]:


from sklearn.dummy import DummyClassifier
freqentist = DummyClassifier(strategy='most_frequent')
freqentist.fit(train_features, train_labels)
print("baseline prediction accuracy is {}".format(freqentist.score(test_features, test_labels).round(4)))


# ### Try multivariate linear regression

# In[59]:


ols = linear_model.LinearRegression(normalize = False)
model = ols.fit(train_features, train_labels)


# In[60]:


print("OLS regression has an R^2 of {} in sample".format(model.score(train_features, train_labels).round(4)))
print("OLS regression has an R^2 of {} out of sample".format(model.score(test_features, test_labels).round(4)))


# 
# ####  lets work harder on variable selection

# In[61]:


varSelection = modelDf[["application_type", "collections_12_mths_ex_med", "earliest_cr_line", "emp_length",
         "funded_amnt", "grade", "home_ownership", "installment", "int_rate",
         "pub_rec", "revol_util", "total_acc", "grade",
         "acc_now_delinq", "recoveries", "loan_status"]]


# In[62]:


varSelection = varSelection[(varSelection["loan_status"] == "Fully Paid") |
                                          (varSelection["loan_status"] == "Charged Off") |
                                          (varSelection["loan_status"] == "Default")]
statusDummies = pd.get_dummies(varSelection["loan_status"])
varSelection["defaulted"] = (statusDummies[["Fully Paid"]] * - 1) + 1


# In[63]:


applicationDummies = pd.get_dummies(varSelection[["application_type", "home_ownership"]])
varSelection = varSelection.drop(columns = ["application_type", "home_ownership"])
applicationDummies = applicationDummies.drop(columns = ["application_type_INDIVIDUAL"])
varSelection = applicationDummies.join(varSelection)


# In[64]:


varSelection["earliest_cr_line"] = varSelection["earliest_cr_line"].str.extract("\w{3}-(\d{4})").astype(float)
varSelection["emp_length"] = varSelection["emp_length"].str.extract("(\d+).*").astype(float)


# In[65]:


varSelection = varSelection.dropna()


# In[66]:


labels = varSelection["defaulted"]
features = varSelection.drop(columns=["defaulted", "loan_status"])
features.info()


# In[67]:


from sklearn.feature_selection import SelectKBest
#take 5 strongest variables
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=5).fit_transform(features, labels)
features = X_new


# In[68]:


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.8, random_state = 42)


# In[69]:


ols = linear_model.LinearRegression(normalize = False)
model = ols.fit(train_features, train_labels)


# In[70]:


print("OLS regression has an R^2 of {} in sample".format(model.score(train_features, train_labels).round(4)))
print("OLS regression has an R^2 of {} out of sample".format(model.score(test_features, test_labels).round(4)))


# In[71]:


from sklearn.linear_model import Lasso

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
res = clf.fit(train_features, train_labels)
score = res.score(test_features, test_labels)
print("LASSO regression has a score of {} out of sample".format(score.round(4)))


# #### let's be more strict about features - rank and remove

# In[72]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(features, labels.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
rfe.score(test_features, test_labels)


# ### remove a few features according to rfe results

# In[73]:


labels = varSelection["defaulted"]
features = varSelection.drop(columns=["defaulted", "loan_status", "application_type_JOINT", "home_ownership_ANY", "home_ownership_NONE"])
features.info()


# In[74]:


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.8, random_state = 42)


# In[75]:


ols = linear_model.LinearRegression(normalize = False)
model = ols.fit(train_features, train_labels)


# In[76]:


print("OLS regression has an R^2 of {} in sample".format(model.score(train_features, train_labels).round(4)))
print("OLS regression has an R^2 of {} out of sample".format(model.score(test_features, test_labels).round(4)))


# #### alas the results are similar.

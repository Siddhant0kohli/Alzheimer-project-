#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


file_path = '/Users/siddhant/Downloads/alzheimers_disease_data.csv'
primary_data = pd.read_csv(file_path)
primary_data.head()


# **Data Exploration / Cleaning**

# In[3]:


primary_data.info()


# In[4]:


primary_data.PatientID.nunique() #All are unique patients 


# In[5]:


float_columns = primary_data.select_dtypes(include = ['float64'])


# In[6]:


float_columns


# In[7]:


primary_data.isna().sum() 


# **Findings so far**
# 
# .No duplicated records
# 
# .No Na records
# 
# .No Null records
# 
# .The data is label encoded already

# In[8]:


primary_data.drop(columns = 'DoctorInCharge', inplace = True)


# In[9]:


primary_data


# In[10]:


primary_data['Age'].head()


# In[11]:


plt.figure(figsize=(10, 6))
sns.histplot(primary_data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[12]:


from scipy.stats import skew, kurtosis


# In[13]:


age_data = primary_data['Age'] 

age_skewness = skew(age_data)
age_kurtosis = kurtosis(age_data)

print(f"Skewness of age: {age_skewness}")
print(f"Kurtosis of age: {age_kurtosis}")


# In[ ]:





# In[14]:


gender_counts = primary_data['Gender'].value_counts()
print(gender_counts)

#Almost Balanced Gender sample no require of Undersampling/Oversampling


# In[15]:


plt.figure(figsize=(10, 6))
sns.histplot(primary_data['Gender'].dropna(), bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[16]:


ethnicity_counts = primary_data['Ethnicity'].value_counts()
ethnicity_counts

#•	Ethnicity: The ethnicity of the patients, coded as follows:
#•	0: Caucasian
#•	1: African American
#•	2: Asian
#•	3: Other


# In[17]:


columns = primary_data.select_dtypes(include=['float64']).columns


for column in columns:
    plt.figure(figsize=(8, 5))  
    plt.hist(primary_data[column], bins= 30)  
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[18]:


columns = primary_data.select_dtypes(include=['float64']).columns

results = ""


for column in columns:
    column_skewness = skew(primary_data[column])
    column_kurtosis = kurtosis(primary_data[column])
    
    results += f"Skewness of {column}: {column_skewness:.4f}\n"
    results += f"Kurtosis of {column}: {column_kurtosis:.4f}\n"
    results += '\n'  

print(results)


# In[19]:


columns = primary_data.select_dtypes(include=['float64']).columns

fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(40, 5))

for i, column in enumerate(columns):
    axes[i].boxplot(primary_data[column], patch_artist=True, showmeans=True)
    axes[i].set_title(f'Boxplot of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Values')
    axes[i].grid(True)

plt.tight_layout()
plt.show()


# **Checking for outliers**

# In[20]:


#Z score method
def z_score_scaling(df, column): 
    mean_column = primary_data[column].mean()
    std_column = primary_data[column].std()
    
    return (df[column] - mean_column) / std_column


# In[21]:


scale_column = primary_data.select_dtypes(include=['float64']).columns
scale_column 


# In[22]:


scaled_columns = z_score_scaling(primary_data, scale_column)
scaled_columns


# In[23]:


for column in scaled_columns:
    z_scores = z_score_scaling(primary_data, column)
    has_outliers = False
    for zscore in z_scores:
        if zscore < -3 or zscore > 3:
            print(f"Outlier in column '{column}': {zscore}")
            has_outliers = True
    if not has_outliers:
        print(f"No outliers found in column '{column}'.")


# In[24]:


#Tukey method
def tukey_outliers(primary_data, column):
    Q1 = primary_data[column].quantile(0.25)
    Q3 = primary_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return primary_data[(primary_data[column] < lower_bound) | (primary_data[column] > upper_bound)]


# In[25]:


tukey_scaled_columns = tukey_outliers(primary_data, scale_column)
tukey_scaled_columns


# In[26]:


numeric_columns = primary_data.select_dtypes(include=['float64']).columns


for column in numeric_columns:
    outliers = tukey_outliers(primary_data, column)
    if not outliers.empty:
        print(f"Outliers in column '{column}':")
        print(outliers)
    else:
        print(f"No outliers found in column '{column}'.")


# **Proven we have no outliers**

# # Normality test

# In **Shapiro-Wilk Test** large P value does not prove that the distribution is normal - It is just not significantly different from normal

# In[27]:


from scipy.stats import shapiro


normality_data_BMI = primary_data['BMI']


stat, p_value = shapiro(normality_data_BMI)

print('Shapiro-Wilk Test:')
print(f'Statistic: {stat}')
print(f'P-value: {p_value}')


# In[28]:


normality_data_Age = primary_data['Age']


stat, p_value = shapiro(normality_data_Age)

print('Shapiro-Wilk Test:')
print(f'Statistic: {stat}')
print(f'P-value: {p_value}')


# In[29]:


normality_data_ADL = primary_data['ADL']


stat, p_value = shapiro(normality_data_ADL)

print('Shapiro-Wilk Test:')
print(f'Statistic: {stat}')
print(f'P-value: {p_value}')


# In[30]:


normality_data_AC = primary_data['AlcoholConsumption']


stat, p_value = shapiro(normality_data_AC)

print('Shapiro-Wilk Test:')
print(f'Statistic: {stat}')
print(f'P-value: {p_value}')


# In[31]:


normality_data_PA = primary_data['PhysicalActivity']


stat, p_value = shapiro(normality_data_PA)

print('Shapiro-Wilk Test:')
print(f'Statistic: {stat}')
print(f'P-value: {p_value}')


# In[32]:


normality_data_DQ = primary_data['DietQuality']


stat, p_value = shapiro(normality_data_DQ)

print('Shapiro-Wilk Test:')
print(f'Statistic: {stat}')
print(f'P-value: {p_value}')


# **It is concluded that variables are normally distributed**
# 
# We can also run kolmogorov - Smirnov test , But to reducd the complexity and scope of this project I will assume normality and assume due to less number of samples the distribution tends to normal compiling with Law of Large numbers. I also tried Box Cox transformation but higher alpha gives slight normal distribution which can cause bias in our predictive model.

# In[33]:


bmi_data = primary_data['BMI']

bmi_skewness = skew(age_data)
bmi_kurtosis = kurtosis(bmi_data)

print(f"Initial Skewness of Bmi: {bmi_skewness}")
print(f"Initial Kurtosis of Bmi: {bmi_kurtosis}")

bmi_data_positive = bmi_data[bmi_data > 0]
bmi_transformed, lambda_value = boxcox(bmi_data_positive)

bmi_transformed_skewness = skew(bmi_transformed)
bmi_transformed_kurtosis = kurtosis(bmi_transformed)

print(f"Transformed Skewness of BMI: {bmi_transformed_skewness}")
print(f"Transformed Kurtosis of BMI: {bmi_transformed_kurtosis}")
print(f"Box-Cox Lambda: {lambda_value}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(bmi_data, kde=True, ax=axes[0])
axes[0].set_title('Original BMI Data')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('Frequency')

sns.histplot(bmi_transformed, kde=True, ax=axes[1])
axes[1].set_title('Box-Cox Transformed BMI Data')
axes[1].set_xlabel('Transformed BMI')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# # Statistical Data Analysis

# **Assuming Normally distributed variables**

# In[34]:


primary_data


# **Practical application of Central Limit Theorem on BMI Variable to verify**

# In[35]:


BMI_sample = np.random.choice(primary_data['BMI'] , size = 200, replace = False)


# In[36]:


BMI_sample


# In[37]:


np.mean(BMI_sample)


# In[38]:


def sample_mean_calculator(sample_array, sample_size, n_sample):
    
    sample_means= []
    for i in range(n_sample):
        sample = np.random.choice(sample_array, size = sample_size , replace = False)
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)
    return sample_means


# In[39]:


sns.displot(sample_mean_calculator(primary_data['BMI'] , 200, 35), kde = True)


# In[40]:


sns.displot(sample_mean_calculator(primary_data['BMI'] , 200, 45)  , kde = True)


# In[41]:


sns.displot(sample_mean_calculator(primary_data['BMI'] , 200, 50), kde = True)


# In[42]:


sns.displot(sample_mean_calculator(primary_data['BMI'] , 200, 700), kde = True)


# **Astonishing visualisation of Central Limit theorem in practice** 

# In[43]:


sns.displot(sample_mean_calculator(primary_data['BMI'] , 600, 1000), kde = True)


# In[44]:


np.mean(sample_mean_calculator(primary_data['BMI'] , 600, 1000))


# In[45]:


numeric_columns = primary_data.select_dtypes(include=['float64']).columns

variable_means = []
for column in numeric_columns:
    data = primary_data[column]
    means = sample_mean_calculator(data, sample_size= 500, n_sample = 100)
    sampled_mean = np.mean(means)
    variable_means.append(sampled_mean)
    print(f"Column: {column}, Sampled Mean: {sampled_mean}")


# In[46]:


primary_data.describe(include = 'float64')


# **Probability tests**

# Q1: What is the probability that a randomly selected patient is Female

# In[47]:


female_count = (primary_data['Gender'] == 1).sum()  
total_count = len(primary_data)  # Total number of patients

prob_female = female_count / total_count * 100
print(f"Probability that a randomly selected patient is Female = {prob_female}")


# In[ ]:





# Q2: Given that a patient is African American, what is the probability that they have a Bachelor's degree or higher?

# In[48]:


african_american_patients = primary_data[primary_data['Ethnicity'] == 1]

bachelors_or_higher = african_american_patients[african_american_patients['EducationLevel'].isin([2, 3])]

prob_bachelors_or_higher = len(bachelors_or_higher) / len(african_american_patients) * 100

print(f"Probability that an African American patient has a Bachelor's degree or higher: {prob_bachelors_or_higher:.2f}%")


# In[ ]:





# Q3:	What is the probability that a patient has a family history of Alzheimer's Disease and also has hypertension?

# In[49]:


family_history_and_hypertension = primary_data[(primary_data['FamilyHistoryAlzheimers'] == 1) & (primary_data['Hypertension'] == 1)]

prob_family_history_and_hypertension = len(family_history_and_hypertension) / len(primary_data) * 100

print(f"Probability that a patient has a family history of Alzheimer's Disease and also has hypertension: {prob_family_history_and_hypertension:.2f}%")


# In[ ]:





# Q4: If a patient has depression, what is the probability that they also have diabetes?

# In[50]:


patient_with_dep = primary_data[(primary_data['Depression']== 1)]

patient_with_diab = patient_with_dep[(patient_with_dep['Diabetes'] == 1)]

patient_with_dep_and_diab = len(patient_with_diab) / len(patient_with_dep) * 100

print(f"Probability that a patient with depression also has diabetes: {patient_with_dep_and_diab:.2f}%")


# In[ ]:





# In[51]:


primary_data['SystolicBP'].describe()


# In[52]:


sns.displot(primary_data['SystolicBP'], kde = True)


# In[53]:


patients_in_range = primary_data[(primary_data['SystolicBP'] >= 120) & (primary_data['SystolicBP'] <= 140)]

prob_systolic_120_140 = len(patients_in_range) / len(primary_data) *100

print(f"Probability that a patient has systolic blood pressure between 120 and 140 mmHg: {prob_systolic_120_140:.2f}%")


# In[ ]:





# Q5: Given that a patient has LDL cholesterol levels above 130 mg/dL, what is the probability that their HDL cholesterol levels are below 40 mg/dL?

# In[54]:


ldl_above_130 = primary_data[primary_data['CholesterolLDL'] > 130]

hdl_below_40 = ldl_above_130[ldl_above_130['CholesterolHDL'] < 40]

prob_hdl_below_40_given_ldl_above_130 = len(hdl_below_40) / len(ldl_above_130) * 100

print(f"Probability that HDL cholesterol levels are below 40 mg/dL given LDL cholesterol levels are above 130 mg/dL: {prob_hdl_below_40_given_ldl_above_130:.2f}%")


# In[ ]:





# In[55]:


primary_data


# In[ ]:





# Average BMI of People with depression and Infrential test of comparision between two groups

# In[56]:


Bmi_with_dep = primary_data.groupby('Depression')['BMI'].mean()


# In[57]:


Bmi_with_dep


# In[58]:


len_1_dep = len(primary_data[primary_data['Depression'] == 1])
len_0_dep = len(primary_data[primary_data['Depression'] == 0])
print(f"Number of patients with depression: {len_1_dep}")
print(f"Number of patients with no depression: {len_0_dep}")


# In[ ]:





# In[59]:


def sample_mean_calculator(sample_array, sample_size, n_sample):
    
    sample_means= []
    for i in range(n_sample):
        sample = np.random.choice(sample_array, size = sample_size , replace = False)
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)
    return sample_means


# In[60]:


bmi_depression = primary_data[primary_data['Depression'] == 1]['BMI']
bmi_no_depression = primary_data[primary_data['Depression'] == 0]['BMI']

sample_size = 50  
n_sample = 1000

sample_means_depression = sample_mean_calculator(bmi_depression, sample_size, n_sample)
sample_means_no_depression =sample_mean_calculator(bmi_no_depression, sample_size, n_sample)

avg_smd = np.mean(sample_means_depression)
avg_smnd= np.mean(sample_means_no_depression)

print(f"Mean of sampled BMI of patients with depression: {avg_smd:.2f}")
print(f"Mean of sampled BMI of patients with no depression: {avg_smnd:.2f}")

std_original_depression = np.std(bmi_depression, ddof=1)
std_original_no_depression = np.std(bmi_no_depression, ddof=1)



print(f"STD of sampled BMI of patients with depression: {std_original_depression:.2f}")
print(f"STD of sampled BMI of patients with no depression: {std_original_no_depression:.2f}")

print(f"Number of patients with depression: {len_1_dep}")
print(f"Number of patients with no depression: {len_0_dep}")


# **To assume equal or unequal variances we will perform F test first**

# The independent t-test assumes the variances of the two groups you are measuring are equal in the population. If your variances are unequal, this can affect the Type I error rate.

# In[61]:


var_nodep = std_original_no_depression ** 2
var_dep = std_original_depression ** 2


# In[62]:


print("Ho: Sampled variances have no diffrence")
print("Ha: Sampled variances have a significant diffrence\n")


print(f"STD of sampled BMI of patients with depression: {std_original_depression:.2f}")
print(f"STD of sampled BMI of patients with no depression: {std_original_no_depression:.2f}\n")

dof_vardep = len_1_dep - 1
dof_varnodep = len_0_dep - 1

print(f"N-1 of Larger variance: {dof_vardep}")
print(f"N-1 of Smaller variance: {dof_varnodep}\n")

f_statistic = var_nodep ** 2 /  var_dep ** 2

print(f"F-statistic: {f_statistic:.3f}")

los = 0.05

tabular_value = 1.13 


if f_statistic > tabular_value:
    print("Ho rejected and there is a significant diffrence between two variance")
    
else:
    print("Ho will be accepted and there is no significant diffrence between the variance")


# Now since we do not have any significant diffrence in the variances we can compare the means with equal variances 

# In[ ]:





# **Ho: µ1 = µ2 (The mean BMI of patients with depression is equal to Mean BMI of patients with no depression)**
# 
# **Ha: µ1 ≠ µ2 (The two BMI means are not equal)**

# **Significance Level**
# 
# Alpha = 0.05 or 5%
# 
# Level of significance = 95%

# In[63]:


avg_smd = np.mean(sample_means_depression)
avg_smnd= np.mean(sample_means_no_depression)

print(f"Mean of sampled BMI of patients with depression: {avg_smd:.2f}")
print(f"Mean of sampled BMI of patients with no depression: {avg_smnd:.2f}")

print()  

var_original_depression = np.var(bmi_depression, ddof=1)
var_original_no_depression = np.var(bmi_no_depression, ddof=1)

print(f"STD of sampled BMI of patients with depression: {var_original_depression:.2f}")
print(f"STD of sampled BMI of patients with no depression: {var_original_no_depression:.2f}")

print()

n1 = len_1_dep
n2 = len_0_dep


# **Since the F test proved the variances are equal we will use the pooled variance formula to compare the means of two groups**

# In[64]:


pooled_variance = ((n1 - 1) * var_original_depression + (n2 - 1) * var_original_no_depression) / (n1 + n2 - 2)
pooled_variance


# **T- Test**

# In[65]:


t_stat = (avg_smd - avg_smnd) / np.sqrt(pooled_variance * (1/n1 + 1/n2))
t_stat


# In[66]:


df = n1 + n2 - 2


# In[67]:


p_value = stats.t.sf(np.abs(t_stat), df) * 2 


# In[68]:


t_val = abs(1.96107)


# In[69]:


alpha = 0.05
if p_value < alpha:
    print("Ho rejected: There is a significant difference between the means and we will reject the Ho")
else:
    print("Ho accepted: There is no significant difference between the means so we fail to reject the Ho")


# Surprisingly there is no significant diffrence between the average BMI of people with depression and no depression.

# In[70]:


primary_data


# In[71]:


plt.figure(figsize=(10, 4))
sns.countplot(x='HeadInjury', data= primary_data)
plt.title('Head Injury Distribution')
plt.xlabel('Head Injury')
plt.ylabel('Count')
plt.show()


# In[72]:


plt.figure(figsize=(10, 4))
sns.countplot(x= 'FamilyHistoryAlzheimers', data= primary_data)
plt.title('Family History Alzheimers Distribution')
plt.xlabel('Family History of Alzheimers')
plt.ylabel('Count')
plt.show()


# In[ ]:





# **Biserial correlation between Sleep quality and Dignosis (Continous variable and Discrete variable**

# In[73]:


import numpy as np
from scipy import stats

def biserial_correlation(x, y):
    
    assert len(x) == len(y), "Both variables have the same length"
    assert set(np.unique(y)).issubset({0, 1}), "y is binary (0 or 1)"
    
    x = np.array(x)
    y = np.array(y)
    
    
    mean_x1 = np.mean(x[y == 1])
    mean_x0 = np.mean(x[y == 0])
    
    
    std_x = np.std(x, ddof=1)
    
    
    p = np.mean(y)
    q = 1 - p
    
    
    y_ordinate = stats.norm.pdf(stats.norm.ppf(p))
    
   
    r_b = ((mean_x1 - mean_x0) / std_x) * (p * q / y_ordinate)
    
    return r_b



# In[74]:


x_sq = primary_data['SleepQuality']
y_d = primary_data['Diagnosis']


# In[75]:


biserial_correlation(x_sq, y_d)

print(f"Biserial correlation between Sleep quality and Diagnosis of Alzheimer is: {biserial_correlation(x_sq, y_d)}")


# In[ ]:





# In[76]:


x_dd = primary_data['Depression']


# In[77]:


biserial_correlation(x_dd, y_d)
print(f"Biserial correlation between Depression and Diagnosis of Alzheimer is: {biserial_correlation(x_dd, y_d)}")


# In[78]:


x_dct = primary_data['DifficultyCompletingTasks']


# In[79]:


biserial_correlation(x_dct, y_d)
print(f"Biserial correlation between Difficulty Completing Tasks and Diagnosis of Alzheimer is: {biserial_correlation(x_dct, y_d)}")


# In[80]:


x_bp = primary_data['BehavioralProblems']


# In[81]:


biserial_correlation(x_bp, y_d)
print(f"Biserial correlation between Difficulty Behavioral Problems and Diagnosis of Alzheimer is:{biserial_correlation(x_bp, y_d)}")


# In[82]:


primary_data


# # Chi Square Test

# We will run a Chi Square test to see if there is an association between Depression and Dignosis of Alzhemier

# **Null Hypothesis**: The Depression is independent to the diagnosis of Alzhemier
# 
# **Alternate Hypothesis**: The Depression is dependent to the diagnosis of Alzhemier

# In[83]:


Alpha_chi = 0.05
contingency_table_chi = pd.crosstab(primary_data['Diagnosis'], primary_data['Gender'])
contingency_table_chi


# In[84]:


chi2, p, dof, expected = chi2_contingency(contingency_table_chi)


# In[85]:


print("\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)


# In[86]:


if p > Alpha_chi:
    print("Accept the null hypothesis , There is no significant independency between depression and dignosis of Alzhemier")
else:
    print("Reject the null hypothesis , There is a significant dependency between depression and dignosis of Alzhemier")


# In[ ]:





# # Correlation analysis

# In[87]:


#General correlation including all variables and not only continous variables

corr_matrix = primary_data.corr()
corr_matrix


plt.figure(figsize=(35, 50))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# # Predictive analytics

# In[ ]:





# In[88]:


primary_data.drop(columns = 'PatientID', inplace = True)


# In[89]:


primary_data


# In[90]:


primary_data.info()


# # Feature selection

# In[91]:


primary_data


# In[92]:


x = primary_data.iloc[:, :-1]
y = primary_data['Diagnosis']


# In[93]:


from sklearn.feature_selection import mutual_info_classif
importance = mutual_info_classif(x, y)

feature_importance = pd.DataFrame({'feature': x.columns, 'importance': importance})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance) 


# In[94]:


features = feature_importance['feature']
importance_scores = feature_importance['importance']

# Create a bar chart
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.barh(features, importance_scores, color='orange')  
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()


# In[95]:


model_df = primary_data[['FunctionalAssessment', 'ADL','MMSE','MemoryComplaints','BehavioralProblems', 'AlcoholConsumption','CholesterolHDL','SystolicBP','Disorientation','Confusion','SleepQuality','FamilyHistoryAlzheimers', 'Diagnosis' ]]


# In[96]:


model_df


# In[97]:


scaler = MinMaxScaler()


# In[98]:


for col in model_df.columns:
    if pd.api.types.is_numeric_dtype(model_df[col]):
        # Reshape the data for the scaler (required to be 2D)
        model_df[col] = scaler.fit_transform(model_df[[col]])


# In[99]:


model_df


# In[100]:


X = model_df.iloc[:,:-1]
y = model_df['Diagnosis']


# In[101]:


feature_names = ['FunctionalAssessment', 'ADL','MMSE','MemoryComplaints','BehavioralProblems', 'AlcoholConsumption','CholesterolHDL','SystolicBP','Disorientation','Confusion','SleepQuality','FamilyHistoryAlzheimers', 'Diagnosis' ]
class_names = ['Diagnosis']


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# **We have performed feature scaling and appropriate features have been chosen, Hence we are prepared for our Model building**

# 1):- **KNN model**

# In[209]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score


# In[222]:


knn = KNeighborsClassifier(n_neighbors= 8)

knn.fit(X_train, y_train)


# In[178]:


y_pred = knn.predict(X_test)
y_train_knn = knn.predict(X_train)
test_knn_accuracy = accuracy_score(y_test, y_pred)
train_knn_accuracy = accuracy_score(y_train, y_train_knn)

print(f'Test accuracy for knn: {test_knn_accuracy * 100:.2f}%')
print(f'Train accuracy for knn: {train_knn_accuracy * 100:.2f}%')


# In[179]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()


plt.show()


# In[180]:


knn_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(knn_report)


# In[181]:


plt.figure(figsize=(10, 4))
sns.countplot(x= 'Diagnosis', data= primary_data)
plt.title('Diagnosis Distribution')
plt.xlabel('Alzheimers Diagnosis')
plt.ylabel('Count')
plt.show()


# 2:- **Logistic Regression**

# In[109]:


from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[110]:


X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y, test_size=0.3, random_state=42)


# In[185]:


logreg = LogisticRegression(solver='saga', max_iter=100)
logreg.fit(X_train_log, y_train_log)

y_pred_log = logreg.predict(X_test_log)

y_train_pred_log = logreg.predict(X_train_log)

train_log_accuracy = accuracy_score(y_train_log, y_train_pred_log)

test_log_accuracy = accuracy_score(y_test_log, y_pred_log)
print(f"Initial Test Set Accuracy: {test_log_accuracy * 100:.4f}%")
print(f"Initial train Set Accuracy: {train_log_accuracy * 100:.4f}%")


# Now I will play around with parameters and Hyper tune the model

# In[186]:


parameters = {
    'C': [0.1, 1, 10, 100],               # Regularization strength
    'solver': ['liblinear', 'saga'],       # Algorithm to use in the optimization problem
    'penalty': ['l1', 'l2'],               # Norm used in penalization (l1 or l2)
    'max_iter': [100, 200, 300]            # Maximum number of iterations
}


grid_search = GridSearchCV(LogisticRegression(), parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_logreg = grid_search.best_estimator_
best_logreg.fit(X_train, y_train)

y_pred_tuned = best_logreg.predict(X_test)

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned Test Set Accuracy: {tuned_accuracy:.4f}")


# In[187]:


logis_report = classification_report(y_test_log, y_pred_log)
print("Classification Report:")
print(logis_report)


# **My logistic regression equation is:-**

# In[188]:


coefficients = logreg.coef_
intercept = logreg.intercept_

print("Coefficients (weights) for each class:\n", coefficients)
print("Intercept (bias term) for each class:\n", intercept)


# 3:- **Decision Tree model**

# In[115]:


from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.model_selection import cross_val_score
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, test_size=0.3, random_state=42)


# In[221]:


dtree = DecisionTreeClassifier(random_state=42,  max_depth = 7, criterion = 'entropy') # Since our dataset is not that big max depth , Post pruning it to 7 is the optimal solution

dtree.fit(X_train_dt, y_train_dt)


# In[189]:


y_pred_dt = dtree.predict(X_test_dt)

y_train_pred_dt = dtree.predict(X_train_dt)

accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)

train_dt_accuracy = accuracy_score(y_train_dt, y_train_pred_dt)

print(f"Test Set Accuracy: {accuracy_dt*100:.4f}%")
print(f"Train Set Accuracy: {train_dt_accuracy*100:.4f}%")

print("\nClassification Report:\n", classification_report(y_test_dt, y_pred_dt))


# In[193]:


plt.figure(figsize=(40,25))
plot_tree(dtree, filled=True)
plt.show()


# Applying **Cross validation**

# In[194]:


cv_scores = cross_val_score(dtree, X, y, cv= 10, scoring='accuracy')

# Print the cross-validation scores and the average score
print("Cross-Validation Scores: ", cv_scores)
print(f"Average Cross-Validation Score: {cv_scores.mean():.4f}")


# 4:- **SVM Model**

# In[129]:


from sklearn.svm import SVC
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.3, random_state=42)


# In[195]:


svm_model = SVC(kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001)


# In[216]:


svm_model.fit(X_train_svm, y_train_svm)

svm_y_pred = svm_model.predict(X_test_svm)

svm_y_train_pred = svm_model.predict(X_train_svm)


# In[211]:


svm_test_accuracy = accuracy_score(y_test_svm, svm_y_pred)
svm_train_accuracy = accuracy_score(y_train_svm, svm_y_train_pred)

# Print accuracies
print(f"Test Set Accuracy: {svm_test_accuracy * 100:.2f}%")
print(f"Training Set Accuracy: {svm_train_accuracy * 100:.2f}%")


# In[220]:


print(classification_report(y_test_svm, svm_y_pred))


# In[212]:


svm_y_proba = svm_model.decision_function(X_test_svm)  


fpr, tpr, _ = roc_curve(y_test_svm, svm_y_proba)
roc_auc = roc_auc_score(y_test_svm, svm_y_proba)


# In[213]:


plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# # Model conclusions

# **Logistic Regression:**
# **Train Accuracy:** 85.50%
# **Test Accuracy:** 82.30%
# **Difference:** 3.20%
# Interpretation: Logistic regression has a relatively close train and test accuracy, with a small difference, suggesting that the model generalizes well to new data. This indicates a good balance between bias and variance.

# **SVM Classifier:**
# **Train Accuracy:** 85.90%
# **Test Accuracy:** 82.71%
# **Difference:** 3.19%
# Interpretation: The SVM classifier also shows a small difference between train and test accuracy, similar to logistic regression. This suggests that the model generalizes well and is likely robust to overfitting.

# **Conclusion** :-
# Based on the data, Logistic Regression or SVM Classifier would be the best choices due to their good balance between accuracy and generalization, with SVM slightly edging out Logistic Regression because of its slightly higher test accuracy and marginally lower overfitting.

# In[ ]:





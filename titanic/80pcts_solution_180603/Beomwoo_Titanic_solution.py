
# coding: utf-8

# In[1]:


import pandas as pd

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')


# PassengerId 는 필요없다.
# Sex 는 남/여 로 구분, 의미있는 정보라고 생각.
# Nam 은 어떻게 나눌 수 있을까?
# Pclass, Cabin 좌석별로 살 수 있는 차이가 확연히 날 것이라고 판단된다.
# -> 과연? 침몰시간에 사람들이 모두 자신들의 좌석에 있었다는 확신이 부족하다. 침몰시간을 살펴보자.
# Fare? 흠
# Embarked 어디서 탔는지.. 충분히 고려해볼만한 정보일 것
# 남은건, Age, SibSp, Parch

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() #setting seaborn default for plots


# ### age에서 nan값을 채워줘야하는데, nan값을 가진 사람들의 특징을 고려해보아야 하지 않을까?

# In[3]:


age_nan_rows = train[train['Age'].isnull()]


# In[4]:


age_nan_rows.head()


# ### 먼저 가장 간단한 성별을 0,1로 표시

# In[5]:


from sklearn.preprocessing import LabelEncoder
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])
test['Sex'] = LabelEncoder().fit_transform(test['Sex'])


# In[6]:


train.head(10)


# In[7]:


test.head()


# ### 이름의 뒷부분을 고려하기엔 케이스가 너무 많아진다. 이름에서 앞의 성만 따서 생각해보자.

# In[8]:


train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train['Name'].unique()
titles
test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_titles = test['Name'].unique()
test_titles


# In[9]:


titles


# In[10]:


test_titles


# ### 나이의 nan값을 채워주기 위해서, 이름의 성별로 중간값을 구하자.
# 남자별, 여자별로 나누어도 좋지만 성별로 하는 것이 더 정확하지 않을까?
# 
# 하지만 성별로 나누는 것도 정확한 기준이 부족하다. 해당 부분에 대해서 좀 더 생각해볼 필요가 있다.

# In[11]:


train['Age'].fillna(-1, inplace=True)
test['Age'].fillna(-1, inplace=True)

medians = dict()
for title in titles:
    median = train.Age[(train["Age"] != -1) & (train['Name'] == title)].median()
    medians[title] = median


# In[12]:


for index, row in train.iterrows():
    if row['Age'] == -1:
        train.loc[index, 'Age'] = medians[row['Name']]

for index, row in test.iterrows():
    if row['Age'] == -1:
        test.loc[index, 'Age'] = medians[row['Name']]

train.head()


# In[13]:


medians


# In[14]:


test.isnull().sum()


# In[15]:


train.isnull().sum()


# In[16]:


test.head()


# In[17]:


test.head()


# In[18]:


test.isnull().sum()


# In[19]:


train.isnull().sum()


# In[20]:


test_age_nan_rows = test[test['Age'].isnull()]


# In[21]:


test_age_nan_rows


# In[22]:


train.head()


# ### 각 이름별로 산사람과 죽은 사람을 비교해보자.

# In[23]:


fig = plt.figure(figsize=(15,6))

i=1
for title in train['Name'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Title : {}'.format(title))
    train.Survived[train['Name'] == title].value_counts().plot(kind='pie')
    i += 1


# ### 각 이름별로 많이 죽은 성 -> 적게 죽은 성 순서로 값을 매기자

# In[24]:


title_replace = {
    'Don':0,
    'Rev':0,
    'Capt':0,
    'Jonkheer':0,
    'Mr':1,
    'Dr':2,
    'Major':3,
    'Col':3,
    'Master':4,
    'Miss':5,
    'Mrs':6,
    'Mme':7,
    'Ms':7,
    'Lady':7,
    'Sir':7,
    'Mlle':7,
    'the Countess':7
}


# In[25]:


train['Name'].unique()


# ### test에는 Dona라는 성이 있지만, train에는 없다.

# In[26]:


test['Name'].unique()


# In[27]:


test[test['Name'] == 'Dona']


# In[28]:


train['Name'] = train['Name'].apply(lambda x: title_replace.get(x))


# In[29]:


train.head()


# In[30]:


test['Name'] = test['Name'].apply(lambda x: title_replace.get(x))


# In[31]:


test.isnull().sum()


# In[32]:


test[test['Name'].isnull()]


# ### Dona성을 가진 사람은 해당 성별에서 name에 대한 중간값을 넣어주자.

# In[33]:


test[test['Sex'] == 0]['Name'].mean()


# In[34]:


train[train['Sex'] == 0]['Name'].mean()


# In[35]:


test[test['Name'].isnull()]['Sex']


# In[36]:


test[test['Name'].isnull()]['Name']


# In[37]:


test['Name'] = test['Name'].fillna(value=train[train['Sex'] == 0]['Name'].mean())


# In[38]:


test.head()


# In[39]:


test.isnull().sum()


# In[40]:


train.isnull().sum()


# In[41]:


train_test_data = [train, test]


# ### 나이를 세분화하자, 10살부터 5살 단위로 60살까지

# In[42]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age']<=10, 'Age'] = 0,
    dataset.loc[(dataset['Age']>10)&(dataset['Age']<=16), 'Age'] = 1,
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=20), 'Age'] = 2,
    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=26), 'Age'] = 3,
    dataset.loc[(dataset['Age']>26)&(dataset['Age']<=30), 'Age'] = 4,
    dataset.loc[(dataset['Age']>30)&(dataset['Age']<=36), 'Age'] = 5,
    dataset.loc[(dataset['Age']>36)&(dataset['Age']<=40), 'Age'] = 6,
    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=46), 'Age'] = 7,
    dataset.loc[(dataset['Age']>46)&(dataset['Age']<=50), 'Age'] = 8,
    dataset.loc[(dataset['Age']>50)&(dataset['Age']<=60), 'Age'] = 9,
    dataset.loc[ dataset['Age']>60, 'Age'] = 10


# ### 나눈 나이에 대해 나이별로 죽은사람과 산사람 비율을 확인하자

# In[43]:


fig = plt.figure(figsize=(15,6))

i=1
for age in train['Age'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Age : {}'.format(age))
    train.Survived[train['Age'] == age].value_counts().plot(kind='pie')
    i += 1


# ### 이 또한 name과 같이 값을 매기자

# In[44]:


age_point_replace = {
    0: 8,
    1: 6,
    2: 2,
    3: 4,
    4: 1,
    5: 7,
    6: 3,
    7: 2,
    8: 5,
    9: 4,
    10: 0
    
}


# In[45]:


for dataset in train_test_data:
    dataset['age_point'] = dataset['Age'].apply(lambda x: age_point_replace.get(x))


# In[46]:


train.head()


# In[47]:


train.head()


# In[48]:


test.head()


# In[49]:


train.head()


# ### Embarked가 nan인 사람은 S로 채워주자. 이를 보다 근거있는 값으로 채울 수는 없는지 고민해보자.

# In[50]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[51]:


embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[52]:


train.head()


# In[53]:


train.head()


# ### SibSp와 Parch값을 이용하여 FamilySize를 추가한다.

# In[54]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# ### FamilySize가 4보다 큰 남자는 아버지일 확률이 높다.
# 사회적으로 아버지는 자신의 가족을 먼저 살리려고 할 것이라 생각했다.
# 
# 여기에 나이도 추가하면 더 좋을수도 있다.

# In[55]:


maybe_dad_mask = (train['FamilySize'] > 4) & (train['Sex'] == 1)


# In[56]:


maybe_dad_mask.head()


# In[57]:


train['maybe_dad'] = 1


# In[58]:


train.head()


# In[59]:


train.loc[maybe_dad_mask,'maybe_dad'] = 0


# In[60]:


train[train['maybe_dad'] == 0].head()


# ### 그럼 아버지로 분류된 사람들이나 아버지가 아닌 사람으로 분류된 사람들의 생존비율은 어떻게 될까?

# In[61]:


# fig = plt.figure(figsize=(15,6))

# i=1
# for title in train['Name'].unique():
#     fig.add_subplot(3, 6, i)
#     plt.title('Title : {}'.format(title))
#     train.Survived[train['Name'] == title].value_counts().plot(kind='pie')
#     i += 1
fig = plt.figure()
ax1 = train.Survived[train['maybe_dad'] == 1].value_counts().plot(kind='pie')


# In[62]:


ax2 = train.Survived[train['maybe_dad'] == 0].value_counts().plot(kind='pie')


# In[63]:


test['maybe_dad'] = 1
test_maybe_dad_mask = (test['FamilySize'] > 4) & (test['Sex'] == 1)
test.loc[test_maybe_dad_mask,'maybe_dad'] = 0


# In[64]:


test.head()


# ### FamilySize에 대해서도 새롭게 값을 매기자. 그러기 위해 FamilySize별로 생존비율 확인

# In[65]:


train['FamilySize'].unique()


# In[66]:


test['FamilySize'].unique()


# In[67]:


fig = plt.figure(figsize=(15,6))

i=1
for size in train['FamilySize'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Size : {}'.format(size))
    train.Survived[train['FamilySize'] == size].value_counts().plot(kind='pie')
    i += 1


# In[68]:


size_replace = {
    1: 3,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 1,
    7: 4,
    8: 0,
    11: 0
}


# In[69]:


for dataset in train_test_data:
    dataset['fs_point'] = dataset['FamilySize'].apply(lambda x: size_replace.get(x))
    dataset.drop('FamilySize',axis=1,inplace=True)


# In[70]:


train.head()


# In[71]:


train.isnull().sum()


# In[72]:


test.isnull().sum()


# ### Pclass별로 생존비율 확인후 새로운 값 매기기

# In[73]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Pclass'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Pclass : {}'.format(x))
    train.Survived[train['Pclass'] == x].value_counts().plot(kind='pie')
    i += 1


# In[74]:


for dataset in train_test_data:
    dataset.loc[dataset['Pclass']==3,'Pclass_point'] = 0
    dataset.loc[dataset['Pclass']==2,'Pclass_point'] = 1
    dataset.loc[dataset['Pclass']==1,'Pclass_point'] = 2


# In[75]:


train.head()


# ### Embarked 별로

# In[76]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Embarked'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Em : {}'.format(x))
    train.Survived[train['Embarked'] == x].value_counts().plot(kind='pie')
    i += 1


# In[77]:


for dataset in train_test_data:
    dataset.loc[dataset['Embarked']==0,'Em_point'] = 0
    dataset.loc[dataset['Embarked']==2,'Em_point'] = 1
    dataset.loc[dataset['Embarked']==1,'Em_point'] = 2


# In[78]:


train.head()


# In[79]:


train.isnull().sum()


# ### Cabin은 해당인원의 숙소? 가 어디있는지에 대한 값이다. 하지만 아래와 같이 값이 너무 다양하다. 뒤의 숫자는 빼고 고려해보자

# In[80]:


train['Cabin'].unique()


# ### Cabin이 nan값인 사람들은 U로 채우자. U는 단지 nan값을 나타내는 문자열이다.
# ### 이를 Fare데이터와 비교해서 넣어도 좋을 것 같다. Fare의 nan값은 1개라서 일단 0으로 넣었다.

# In[81]:


for data in train_test_data:
    data['Cabin'].fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
    data['Cabin'].unique()
    data['Fare'].fillna(0,inplace=True)
    data['Fare'] = data['Fare'].apply(lambda x: int(x))


# ### Cabin별로 생존비율 확인

# In[82]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1


# ### Fare의 값도 너무 다양하다. 왜 그러지? 이에 대해서 궁금하다.
# 일단 이를 그대로 처리하는 것은 힘들 것 같다. 이 또한 구간별로 나누어 생각해보자.

# In[83]:


temp = train['Fare'].unique()
temp.sort()
temp


# In[84]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare']<=30, 'Fare'] = 0,
    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=80), 'Fare'] = 1,
    dataset.loc[(dataset['Fare']>80)&(dataset['Fare']<=100), 'Fare'] = 2,
    dataset.loc[(dataset['Fare']>100), 'Fare'] = 3


# ### 아까 Cabin의 값을 U로 넣어준 사람들에게 존재하는 Cabin값으로 넣어주기 위해 각 Cabin별로 어떤 Fare범위가 가장 많은지 확인하고 해당 값으로 U값을 대체한다.

# In[85]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1


# In[86]:


for dataset in train_test_data:
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 0), 'Cabin'] = 'G',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 1), 'Cabin'] = 'T',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 2), 'Cabin'] = 'C',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 3), 'Cabin'] = 'B',


# In[87]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1


# ### Cabin의 값 별로 생존여부 확인

# In[88]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1


# In[89]:


for dataset in train_test_data:
    dataset.loc[(dataset['Cabin'] == 'G'), 'Cabin_point'] = 0,
    dataset.loc[(dataset['Cabin'] == 'C'), 'Cabin_point'] = 3,
    dataset.loc[(dataset['Cabin'] == 'E'), 'Cabin_point'] = 5,
    dataset.loc[(dataset['Cabin'] == 'T'), 'Cabin_point'] = 1,
    dataset.loc[(dataset['Cabin'] == 'D'), 'Cabin_point'] = 7,
    dataset.loc[(dataset['Cabin'] == 'A'), 'Cabin_point'] = 2,
    dataset.loc[(dataset['Cabin'] == 'B'), 'Cabin_point'] = 6,
    dataset.loc[(dataset['Cabin'] == 'F'), 'Cabin_point'] = 4,


# In[90]:


train.head()


# ### Fare별로 생존여부확인 및 점수 대체

# In[91]:


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Fare'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Fare : {}'.format(x))
    train.Survived[train['Fare'] == x].value_counts().plot(kind='pie')
    i += 1


# In[92]:


for dataset in train_test_data:
    dataset.loc[(dataset['Fare'] == 0), 'Fare_point'] = 0,
    dataset.loc[(dataset['Fare'] == 1), 'Fare_point'] = 1,
    dataset.loc[(dataset['Fare'] == 2), 'Fare_point'] = 3,
    dataset.loc[(dataset['Fare'] == 3), 'Fare_point'] = 2,


# In[93]:


# for dataset in train_test_data:
#     dataset['parent'] = 1
#     dataset.loc[(dataset['Parch'] > 0) & (dataset['Age'] >= 4), 'parent'] = 0


# In[94]:


# train.Survived[train['parent'] == 1].value_counts().plot(kind='pie')


# In[95]:


# train.Survived[train['parent'] == 0].value_counts().plot(kind='pie')


# ### 모든 값을 정규화? 한다. 정규화랑 좀 다른 것일지도 모른다. 해당 내용에 대해서는 한번 더 공부가 필요하다.
# 단지 이것을 하지 않았을 때 보다 했을 때 결과가 더 좋았다.. 왜일까?

# In[96]:


from sklearn.preprocessing import StandardScaler
for dataset in train_test_data:
    dataset['Name'] = StandardScaler().fit_transform(dataset['Name'].values.reshape(-1, 1))
    dataset['Sex'] = StandardScaler().fit_transform(dataset['Sex'].values.reshape(-1, 1))
    dataset['maybe_dad'] = StandardScaler().fit_transform(dataset['maybe_dad'].values.reshape(-1, 1))
    dataset['fs_point'] = StandardScaler().fit_transform(dataset['fs_point'].values.reshape(-1, 1))
    dataset['Em_point'] = StandardScaler().fit_transform(dataset['Em_point'].values.reshape(-1, 1))
    dataset['Cabin_point'] = StandardScaler().fit_transform(dataset['Cabin_point'].values.reshape(-1, 1))
    dataset['Pclass_point'] = StandardScaler().fit_transform(dataset['Pclass_point'].values.reshape(-1, 1))
    dataset['age_point'] = StandardScaler().fit_transform(dataset['age_point'].values.reshape(-1, 1))
    dataset['Fare_point'] = StandardScaler().fit_transform(dataset['Fare_point'].values.reshape(-1, 1))


# ### 필요없는 속성 지우기

# In[97]:


train.drop(['PassengerId','Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)
test.drop(['Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)


# In[98]:


train.head()


# In[99]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']


# In[100]:


train_data.head()


# In[101]:


target.head()


# In[102]:


test.shape


# In[103]:


train.shape


# In[104]:


train_data.shape


# ### SVC를 이용하여 결과를 만들었다.

# In[105]:



# Importing Classifier Modules# Import 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = SVC()
# clf = DecisionTreeClassifier()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission_test1.csv', index=False)
submission = pd.read_csv('submission_test1.csv')
submission.head()


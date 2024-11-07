

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#나이에 따른 생존률 계산
titanic = sns.load_dataset('titanic')
print(titanic.info()) # 결축치 존재
#1
titanic_drop_row = titanic.dropna(subset=['age'])
print(titanic.info())
#2 생존률
titanic_drop_row['survived'] = titanic_drop_row['survived'].astype(float)
print(titanic_drop_row['survived'])

plt.figure(figsize=(10,5))

sns.histplot(data=titanic_drop_row, x='age', weights='survived', bins=8, kde=False)
plt.title('Survived Rate by Age (Drop NaN rows)')
plt.xlabel('Age')
plt.ylabel('Survival Rate (weighted)')
plt.show()




# # Titanic dataset 로드
# titanic = sns.load_dataset('titanic')
#
# # 결측치 처리: 나이 컬럼에서 결측치를 제거
# titanic = titanic.dropna(subset=['age'])
#
# # 나이별 생존률 계산
# age_survival_rate = titanic.groupby('age')['survived'].mean()
#
# # 나이별 생존률을 히스토그램으로 시각화
# age_survival_rate.plot(kind='hist', bins=20, edgecolor='black', alpha=0.7)
#
# # 그래프 꾸미기
# plt.title('Age-wise Survival Rate')
# plt.xlabel('Age')
# plt.ylabel('Survival Rate')
# plt.show()







# print(titanic['sex'].head())
# print(titanic.info())
# gender_survival = titanic.groupby(by='sex')['survived'].mean()
# print(type(gender_survival)) #series


# gender_survival = titanic.groupby(by='sex')['survived'].mean().reset_index()
# print(gender_survival)
# # print(type(gender_survival)) #series
# print(gender_survival.info())
#
#
# sns.barplot(data=gender_survival, x='sex', y='survived')
# plt.title('Survival Rate by Gender')
# plt.ylabel('Survival Rate')
# plt.show()








#print(titanic['survived'])
#print(titanic.info())

#생존자 수와 사망자 수 시각화

# sns.countplot(data=titanic, x='survived')
# plt.title('Survived (0 = NO, 1 = Yes)')
# plt.xlabel('Survived')
# plt.ylabel('Count')
# plt.show()



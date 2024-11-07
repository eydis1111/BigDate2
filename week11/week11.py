import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


#print(titanic['survived'])
#print(titanic.info())

#생존자 수와 사망자 수 시각화
titanic = sns.load_dataset('titanic')
sns.countplot(data=titanic, x='survived')
plt.title('Survived (0 = NO, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()



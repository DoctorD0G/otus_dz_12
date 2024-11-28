import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print(f"Количество пассажиров: {df.shape[0]}")
print(f"Доля выживших пассажиров: {df['Survived'].mean():.2f}")

sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Выживаемость в зависимости от пола')
plt.show()

sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title('Распределение возраста пассажиров')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Выживаемость в зависимости от возраста')
plt.xlabel('Выжил (0 - нет, 1 - да)')
plt.ylabel('Возраст')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df, palette='pastel')
plt.title('Выживаемость в зависимости от класса')
plt.xlabel('Класс')
plt.ylabel('Количество пассажиров')
plt.legend(title='Выжил', loc='upper right', labels=['Нет', 'Да'])
plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

sns.histplot(df['Fare'], kde=True, color='green', bins=40)
plt.title('Распределение стоимости билета')
plt.xlabel('Стоимость билета')
plt.ylabel('Количество пассажиров')
plt.show()

sns.barplot(x='Embarked', y='Survived', data=df, ci=None, palette='muted')
plt.title('Выживаемость по порту посадки')
plt.xlabel('Порт посадки')
plt.ylabel('Доля выживших')
plt.show()

df['Relatives'] = df['SibSp'] + df['Parch']
sns.barplot(x='Relatives', y='Survived', data=df, ci=None, palette='viridis')
plt.title('Выживаемость в зависимости от количества родственников')
plt.xlabel('Количество родственников на борту')
plt.ylabel('Доля выживших')
plt.show()


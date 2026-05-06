# preprocess_tools.py
import pandas as pd

def clean_titanic_data(df):
    
    # 1. 敬称（Title）の抽出：名前から "Mr." などを抜き出す(追加)
    # 正規表現（ ' ([A-Za-z]+)\.' ）を使って、スペースとドットの間の文字を抽出
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # 特殊な敬称や表記ゆれを整理
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 敬称をAIが計算できる数字に変換
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 2, "Master": 3, "Rare": 1}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    # 2. 欠損値の補完
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean()) # Day2のメモ通り平均値を使用
    
    # 3. 特徴量エンジニアリング（家族数の統合）
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 必要な列だけを抽出
    features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Title']
    return df[features], df['Survived']
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Preprocessing.preprocessing import clean_text

df = pd.read_csv('data/raw/kaggle/train.csv')

df['clean_text'] = df['text'].apply(clean_text)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)

train_df.to_csv('data/processed/English/train.csv', index=False)
test_df.to_csv('data/processed/English/test.csv', index=False)

print("Dataset has been processed and saved to 'data/processed/English/' directory.")
import pandas as pd
import glob 

from src.Preprocessing.preprocessing import clean_text
from sklearn.model_selection import train_test_split

#====================================================================
# Load and preprocess the CrisisMD dataset for urgency classification
#====================================================================

files = glob.glob('data/raw/crisismd/annotations/*.tsv')

files = [ file for file in files if not file.split("\\")[-1].startswith("_") ]

print("Files Loaded:")

for file in files:
    print(file)

df_list = []

for file in files:
    temp_df = pd.read_csv(file, sep="\t")
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

print("\nColumns :", df.columns)

df = df[["tweet_text", "text_human"]]
df = df.rename(columns={"tweet_text": "text", "text_human": "label"})

df = df.dropna()

df["clean_text"] = df["text"].apply(clean_text)

print("\nUnique Labels :", df["label"].unique())

label_mapping = {
    "not_humanitarian": 0,
    "infrastructure_and_utility_damage": 1,
    "rescue_volunteering_or_donation_effort": 2,
    "affected_individuals": 3
}

df["urgency_label"] = df["label"].map(label_mapping)

df = df.dropna(subset=["urgency_label"])

df["urgency_label"] = df["urgency_label"].astype(int)

df.to_csv("data/processed/urgency/crisismd_urgency.csv", index=False)

print("\nPreprocessing Completed. Processed data saved to 'data/processed/crisismd/crisismd_urgency.csv'")

#=================================================
# Split the dataset into training and testing sets
#=================================================

df = pd.read_csv("data/processed/urgency/crisismd_urgency.csv")

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['urgency_label'], random_state=42)

train_df.to_csv('data/processed/urgency/train.csv', index=False)
test_df.to_csv('data/processed/urgency/test.csv', index=False)

print("Done")
import pandas as pd
from deep_translator import GoogleTranslator

train_df = pd.read_csv("data/processed/english/train.csv")
test_df = pd.read_csv("data/processed/english/test.csv")


def translate_test(text):
    try:
        translated = GoogleTranslator(source='auto', target='id').translate(text)
        return translated
    except:
        return text

train_df['translated_text'] = train_df['clean_text'].apply(translate_test)
test_df['translated_text'] = test_df['clean_text'].apply(translate_test)

train_df.to_csv("data/processed/Indonesian/translated_train.csv", index=False)
test_df.to_csv("data/processed/Indonesian/translated_test.csv", index=False)

print("Translation completed and saved to translated_train.csv")
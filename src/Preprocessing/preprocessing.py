import re
import emoji

def clean_text(text):
    text = str(text)

    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = text.lower()  # Convert to lowercase

    return text
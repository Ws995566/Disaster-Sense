from langdetect import detect

def detect_language(text):
    try:
        language = detect(text)
        
        if language == 'id':
            return 'Indonesian'
        else:
            return 'English'
    
    except:
        return 'Unknown'
import os
import re
import string

TXT_DIR = "data_scrape/txts"
ALL_TXTS_COMBINED = ""

def extract_president_speech(raw_text: str) -> str:
    # if "THE PRESIDENT:" is not there, keep it as it is
    if "THE PRESIDENT:" not in raw_text:
        # Remove unwanted Unicode characters
        cleaned_text = raw_text.replace('\u2028', ' ') 
        cleaned_text = cleaned_text.replace('\u2029', ' ') 

        # Convert to lowercase
        cleaned_text = cleaned_text.lower()

        # Remove Punctuation
        punctuation_to_remove = string.punctuation
        translation_table = str.maketrans('', '', punctuation_to_remove)
        cleaned_text = cleaned_text.translate(translation_table)

        # Multiple white spaces to single white space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Remove "view transcript"
        cleaned_text = cleaned_text.replace("view transcript", "")
        cleaned_text = cleaned_text.replace("Transcript", "")

        return raw_text
    
    # Remove things like "THE PRESIDENT:" or "AUDIENCE MEMBERS:"
    # Also, Remove anything not after "THE PRESIDENT:" (that is, keep only president speeches)
    president_speech_parts = []
    is_president_speaking = False
    speaker_pattern = re.compile(r"^([A-Z][A-Z\s]*):(.*)")
    lines = raw_text.splitlines()

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line: continue
        match = speaker_pattern.match(stripped_line)
        if match:
            speaker_name = match.group(1).strip()
            inline_text = match.group(2).strip()
            if speaker_name == "THE PRESIDENT":
                is_president_speaking = True
                if inline_text:
                    president_speech_parts.append(inline_text)
            else:
                is_president_speaking = False
        else:
            if is_president_speaking:
                president_speech_parts.append(stripped_line)

    # Join collected parts
    joined_text = ' '.join(president_speech_parts)

    # Remove unwanted Unicode characters
    cleaned_text = joined_text.replace('\u2028', ' ') 
    cleaned_text = cleaned_text.replace('\u2029', ' ') 

    # Convert to lowercase
    cleaned_text = cleaned_text.lower()

    # Multiple white spaces to single white space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Remove "view transcript"
    cleaned_text = cleaned_text.replace("view transcript", "")

    return cleaned_text

for file in os.listdir(f"{TXT_DIR}"):
    with open(f"{TXT_DIR}/{file}", "r") as f:
        transcript = f.read()
    
    transcript = extract_president_speech(transcript)
    ALL_TXTS_COMBINED += transcript

    with open(f"{TXT_DIR}/{file}", "w") as f:
        f.write(transcript)

with open(f"{TXT_DIR}/all_txts_combined.txt", "w") as f:
    f.write(ALL_TXTS_COMBINED)
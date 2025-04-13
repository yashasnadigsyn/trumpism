import os

TXT_DIR = "data_scrape/txts"
ALL_TXTS_COMBINED = ""

for file in os.listdir(f"{TXT_DIR}"):
    with open(f"{TXT_DIR}/{file}", "r") as f:
        transcript = f.read()
    
    ALL_TXTS_COMBINED += transcript

with open(f"{TXT_DIR}/all_txts_combined.txt", "w") as f:
    f.write(ALL_TXTS_COMBINED)
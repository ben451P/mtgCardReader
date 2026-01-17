# takes in final card json and then cleans them for use in the ocr finteuning

import json
from datetime import datetime

DATA_PATH = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\all_cards.json"
NEW_PATH =  r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\all_new_cards_cleaned.json" # new card defined creatred after 2018

data = []
with open(DATA_PATH, "r",encoding="utf-8") as file:
    reader = json.load(file)
    for line in reader:
        if not line.get("image_uris",0):continue
        url = line["image_uris"]
        if not url.get("png",0):continue
        if not line.get("released_at",0) or datetime.strptime(line.get("released_at",0), "%Y-%m-%d").year < 2018:
            continue
        url = url["png"]
        dp = {
            "image_path":url,
            "text": line["name"],
        }
        data.append(dp)

with open(NEW_PATH,"w") as file:
    file.write(json.dumps(data,indent=2))
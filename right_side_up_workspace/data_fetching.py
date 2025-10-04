import requests
import time
import csv

processed = {}
card_names = []

for i in range(5000):
    card_data = requests.get("https://api.scryfall.com/cards/random").json()

    if not card_data.get("image_uris",0):
        print(f"Skipped {i} step 1")
        continue

    if processed.get(card_data["id"],0):
        print(f"Skipped {i} step 1.5")
        continue
    else:
        processed[card_data["id"]] = 1

    img = card_data["image_uris"]

    if not img.get("png",0):
        print(f"Skipped {i} step 2")
        continue

    img = img["png"]

    try:
        img = requests.get(img,stream=True).content

        with open(f"dataset/data{i}.png","wb") as file:
            file.write(img)

    except:
        print(f"Skipped {i} step 3")
        continue

    card_names.append([card_data["name"],card_data["id"],i])

    time.sleep(2/100)

card_names = sorted(card_names,key=lambda x: x[2])

with open("d_labels.csv","w") as file:
    writer = csv.writer(file)
    writer.writerows(card_names)
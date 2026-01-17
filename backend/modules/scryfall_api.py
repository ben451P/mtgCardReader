import requests
import json

class ScryfallAPI:
    def __init__(self):
        self.data = None

    def __repr__(self):
        return self.data

    def fetch(self, card_name):
        card_name = card_name.lower()
        card_name = card_name.split()
        card_name = "+".join(card_name)
        
        query = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
        data = requests.get(query)
        self.data = data

    def return_result(self):
        return self.data
        # return "Sorry no result yet"
    
    def write_to_json(self,file_name):
        if not self.data:return "No data yet!"
        with open(file_name, "w") as file:
            file.write(json.dumps(self.data.json(), indent=2))
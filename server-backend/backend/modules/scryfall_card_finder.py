import requests

class ScryfallCardFinder:
    def find(card_name: str) -> dict:
        card_name = card_name.lower()
        card_name = card_name.split()
        card_name = "+".join(card_name)
        
        query = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
        data = requests.get(query)

        return data.json()
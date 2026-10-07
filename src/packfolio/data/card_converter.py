import csv
import random

def load_cards(file_path):
    cards = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            cards.append(row)

    return cards

def get_base_cards(cards):
    base_cards = []

    for card in cards:
        card_name = card["card"]

        if "[" not in card_name and "]" not in card_name and "#" in card_name:
            base_cards.append(card)

    return base_cards

def get_card_number(card):
    card_name = card["card"]

    try:
        return int(card_name.split("#")[-1])
    except ValueError:
        return None

def get_base_rookies(cards):
    base_cards = get_base_cards(cards)

    return [
        card for card in base_cards
        if get_card_number(card) is not None
        and get_card_number(card) >= 301
    ]


def get_base_veterans(cards):
    base_cards = get_base_cards(cards)

    return [
        card for card in base_cards
        if get_card_number(card) is not None
        and get_card_number(card) <= 300
    ]

def get_cards_by_parallel(cards, parallel):
    matching_cards = []

    search_term = f"[{parallel.replace('_', ' ').title()}]"

    for card in cards:
        if search_term.lower() in card["card"].lower():
            matching_cards.append(card)

    return matching_cards


def pull_card_by_parallel(cards, parallel):
    matching_cards = get_cards_by_parallel(cards, parallel)

    if not matching_cards:
        return None

    return random.choice(matching_cards)

if __name__ == "__main__":
    cards = load_cards("data/2020_panini_prizm_cards.csv")

    rookies = get_base_rookies(cards)
    veterans = get_base_veterans(cards)
#### WILL BE REMOVED #### 
    print(f"Rookies: {len(rookies)}")

    print(f"Veterans: {len(veterans)}")

    print(f"Total: {len(rookies) + len(veterans)}")

    print("\nSome rookies:")

    for card in rookies[:10]:

        print(card["card"])
import csv
import random


def load_cards(file_path):
    cards = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            cards.append(row)

    return cards


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

    card = pull_card_by_parallel(cards, "orange")

    print(card["card"])
import csv
import random
from packfolio.probabilities.pull_rates_and_distributions import NUMBERED_PARALLELS

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

def pull_random_cards(cards, amount):
    return random.choices(cards, k=amount)

def get_parallel_version(cards, base_card, parallel):
    card_number = get_card_number(base_card)

    search_term = f"[{parallel.replace('_', ' ').title()}]"

    for card in cards:
        if (
            get_card_number(card) == card_number
            and search_term.lower() in card["card"].lower()
        ):
            return card

    return None

def pull_card_by_parallel(cards, parallel):
    matching_cards = get_cards_by_parallel(cards, parallel)

    if not matching_cards:
        return None

    return random.choice(matching_cards)

def is_numbered_card(card):
    card_name = card["card"].lower()

    for parallel in NUMBERED_PARALLELS:
        parallel_name = parallel.replace("_", " ")

        if f"[{parallel_name}]" in card_name:
            return True

    return False

def get_autograph_cards(cards):
    excluded = [
        "no huddle",
        "red shimmer",
        "blue shimmer",
        "green shimmer",
        "neon green pulsar",
        "pink"
    ]

    autographs = []

    for card in cards:
        name = card["card"].lower()

        if "[autograph" not in name:
            continue

        if any(excluded_name in name for excluded_name in excluded):
            continue

        autographs.append(card)

    return autographs

def get_autograph_variants(cards):
    variants = {}

    for card in get_autograph_cards(cards):
        name = card["card"]

        start = name.find("[")
        end = name.find("]")

        if start != -1 and end != -1:
            variant = name[start + 1:end]
            variants[variant] = variants.get(variant, 0) + 1

    return variants

def pull_random_autographs(cards, amount):
    autographs = get_autograph_cards(cards)

    weights = [
        int(card["print_run"])
        for card in autographs
    ]

    return random.choices(
        autographs,
        weights=weights,
        k=amount
    )

def find_card(cards, card_name):
    for card in cards:
        if card["card"] == card_name:
            return card

    return None

if __name__ == "__main__":
    cards = load_cards("data/2020_panini_prizm_cards.csv")

    autographs = get_autograph_cards(cards)

    with_print_run = [
        card for card in autographs
        if card.get("print_run") not in ("", None)
    ]

    print("Total autographs:", len(autographs))
    print("Autographs with print run:", len(with_print_run))

    for card in with_print_run[:30]:
        print(card["card"], "->", card["print_run"])
        
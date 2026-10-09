import random

from packfolio.probabilities.pull_rates_and_distributions import BOX_CONTENTS, NUMBERED_PARALLELS
from packfolio.data.card_converter import (
    load_cards,
    get_base_rookies,
    get_base_veterans,
    pull_random_cards,
    get_parallel_version,
    is_numbered_card,
    pull_random_autographs,
)
cards = load_cards("data/2020_panini_prizm_cards.csv")
rookie_pool = get_base_rookies(cards)
veteran_pool = get_base_veterans(cards)

def pull_numbered_parallel():
    parallels = list(NUMBERED_PARALLELS.keys())
    probabilities = list(NUMBERED_PARALLELS.values())

    return random.choices(
        parallels,
        weights=probabilities,
        k=1
    )[0]

def apply_numbered_parallels(box, cards):
    # Choose 9 unique card slots in the box
    numbered_indices = random.sample(
        range(len(box)),
        BOX_CONTENTS["numbered"]
    )

    for index in numbered_indices:
        base_card = box[index]

        parallel = pull_numbered_parallel()

        parallel_card = get_parallel_version(
            cards,
            base_card,
            parallel
        )

        if parallel_card is not None:
            box[index] = parallel_card

    return box

def apply_silver_parallels(box, cards):
    # Only use slots that are not already numbered
    available_indices = [
        i for i, card in enumerate(box)
        if not is_numbered_card(card)
    ]

    silver_indices = random.sample(
        available_indices,
        BOX_CONTENTS["silver"]
    )

    for index in silver_indices:
        base_card = box[index]

        silver_card = get_parallel_version(
            cards,
            base_card,
            "silver"
        )

        if silver_card is not None:
            box[index] = silver_card

    return box

def apply_autographs(box, cards):
    available_indices = [
        i for i, card in enumerate(box)
        if not is_numbered_card(card)
        and "[silver]" not in card["card"].lower()
        and card.get("type") != "insert"
    ]

    autograph_indices = random.sample(
        available_indices,
        BOX_CONTENTS["autographs"]
    )

    autographs = pull_random_autographs(
        cards,
        BOX_CONTENTS["autographs"]
    )

    for index, autograph in zip(autograph_indices, autographs):
        box[index] = autograph

    return box

def open_box():
    rookies = pull_random_cards(
        rookie_pool,
        BOX_CONTENTS["rookies"]
    )

    veteran_count = 144 - BOX_CONTENTS["rookies"]

    veterans = pull_random_cards(
        veteran_pool,
        veteran_count
    )

    box = rookies + veterans

    box = apply_numbered_parallels(box, cards)
    box = apply_silver_parallels(box, cards)
    box = apply_inserts(box)
    box = apply_autographs(box, cards)

    return box

def create_insert():
    return {
        "card": "Simulated Insert",
        "type": "insert",
        "ungraded": None
    }

def apply_inserts(box):
    available_indices = [
        i for i, card in enumerate(box)
        if not is_numbered_card(card)
        and "[silver]" not in card["card"].lower()
    ]

    insert_indices = random.sample(
        available_indices,
        BOX_CONTENTS["inserts"]
    )

    for index in insert_indices:
        box[index] = create_insert()

    return box
if __name__ == "__main__":
    box = open_box()

    numbered = [
        card for card in box
        if is_numbered_card(card)
    ]

    silvers = [
        card for card in box
        if "[silver]" in card["card"].lower()
    ]

    inserts = [
        card for card in box
        if card.get("type") == "insert"
    ]

    autographs = [
        card for card in box
        if "[autograph" in card["card"].lower()
    ]

    print("Total cards:", len(box))
    print("Numbered:", len(numbered))
    print("Silver:", len(silvers))
    print("Inserts:", len(inserts))
    print("Autographs:", len(autographs))
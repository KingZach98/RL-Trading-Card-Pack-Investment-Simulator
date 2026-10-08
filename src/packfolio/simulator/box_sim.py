import random

from packfolio.probabilities.pull_rates_and_distributions import BOX_CONTENTS, NUMBERED_PARALLELS
from packfolio.data.card_converter import (
    load_cards,
    get_base_rookies,
    get_base_veterans,
    pull_random_cards,
    pull_card_by_parallel,
    get_parallel_version
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

    return box
if __name__ == "__main__":
    box = open_box()

    print(f"Total cards: {len(box)}")

    numbered = [
        card for card in box
        if card["print_run"] != ""
    ]

    print(f"Numbered cards: {len(numbered)}")

    for card in numbered:
        print(card["card"])
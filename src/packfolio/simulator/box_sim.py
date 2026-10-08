import random

from packfolio.probabilities.pull_rates_and_distributions import BOX_CONTENTS, NUMBERED_PARALLELS
from packfolio.data.card_converter import (
    load_cards,
    get_base_rookies,
    get_base_veterans,
    pull_random_cards,
    pull_card_by_parallel,
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


def open_box():
    # box = {
    #     "rookies": BOX_CONTENTS["rookies"],
    #     "silver": BOX_CONTENTS["silver"],
    #     "numbered": [],
    #     "autographs": BOX_CONTENTS["autographs"],
    #     "inserts": BOX_CONTENTS["inserts"],
    # }

    # for _ in range(BOX_CONTENTS["numbered"]):
    #     parallel = pull_numbered_parallel()
    #     card = pull_card_by_parallel(cards, parallel)

    #     box["numbered"].append({
    #         "parallel": parallel,
    #         "card": card
    #     })
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

    return box

if __name__ == "__main__":
    # box = open_box()

    # print("Simulated Hobby Box")
    # print("-------------------")
    # print(f"Rookies: {box['rookies']}")
    # print(f"Silver Prizms: {box['silver']}")
    # print(f"Autographs: {box['autographs']}")
    # print(f"Inserts: {box['inserts']}")
    # print("Numbered Prizms:")

    # for pull in box["numbered"]:
    #     print(f"- {pull['card']['card']}")
    box = open_box()

    print(f"Total cards: {len(box)}")

    for card in box:
        print(card["card"])
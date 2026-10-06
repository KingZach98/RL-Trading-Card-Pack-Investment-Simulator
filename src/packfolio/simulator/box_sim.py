import random

from src.packfolio.probabilities.pull_rates_and_distributions import BOX_CONTENTS, NUMBERED_PARALLELS

def pull_numbered_parallel():
    parallels = list(NUMBERED_PARALLELS.keys())
    probabilities = list(NUMBERED_PARALLELS.values())

    return random.choices(
        parallels,
        weights=probabilities,
        k=1
    )[0]


def open_box():

    box = {
        "rookies": BOX_CONTENTS["rookies"],
        "silver": BOX_CONTENTS["silver"],
        "numbered": [],
        "autographs": BOX_CONTENTS["autographs"],
        "inserts": BOX_CONTENTS["inserts"],
    }

    for _ in range(BOX_CONTENTS["numbered"]):
        parallel = pull_numbered_parallel()
        box["numbered"].append(parallel)

    return box


if __name__ == "__main__":
    box = open_box()

    print("Simulated Hobby Box")
    print("-------------------")
    print(f"Rookies: {box['rookies']}")
    print(f"Silver Prizms: {box['silver']}")
    print(f"Autographs: {box['autographs']}")
    print(f"Inserts: {box['inserts']}")
    print(f"Numbered Prizms: {box['numbered']}")
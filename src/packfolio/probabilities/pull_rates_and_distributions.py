"""
Pull configuration for 2020 Panini Prizm Football Hobby Box.
This file defines the expected box contents and the probability
distributions used when simulating box openings.
"""

# 12 packs per box, 12 cards per pack 
PACKS_PER_BOX = 12
CARDS_PER_PACK = 12
CARDS_PER_BOX = PACKS_PER_BOX * CARDS_PER_PACK

# expected content of whats inside when opening an entire box
BOX_CONTENTS = {
    "rookies": 24,
    "silver_prizms": 2,
    "numbered_prizms": 9,
    "autographs": 2,
    "inserts": 4,
}

# numbered types and their id
NUMBERED_PARALLELS = {
    "orange": 249,
    "blue_wave": 199,
    "hyper": 175,
    "red_wave": 149,
    "blue_ice": 99,
    "green_scope": 75,
    "purple_power": 49,
    "camo": 25,
    "gold": 10,
    "gold_vinyl": 5,
    "black_finite": 1,
}

### NONE FOR NOW ###

NUMBERED_PULL_RATES = {
    "orange": None,
    "blue_wave": None,
    "hyper": None,
    "red_wave": None,
    "blue_ice": None,
    "green_scope": None,
    "purple_power": None,
    "camo": None,
    "gold": None,
    "gold_vinyl": None,
    "black_finite": None,
}
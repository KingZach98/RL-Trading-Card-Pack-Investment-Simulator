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
    "silver": 2,
    "numbered": 9,
    "autographs": 2,
    "inserts": 4,
}

# Approximate pull probabilities for numbered parallels.
# Panini does not provide exact pull odds for each parallel, so the
# distribution is derived from the official serial-numbered print runs.
# Assumes that all parallels have the same checklist size.
# Example: Orange /249 is ~24.0% of the numbered parallel population.

NUMBERED_PARALLELS = {
    'orange': 0.24034749034749034,
    'blue_wave': 0.1920849420849421,
    'hyper': 0.16891891891891891,
    'red_wave': 0.1438223938223938,
    'blue_ice': 0.09555984555984556,
    'green_scope': 0.07239382239382239, 
    'purple_power': 0.0472972972972973, 
    'camo': 0.02413127413127413, 
    'gold': 0.009652509652509652, 
    'gold_vinyl': 0.004826254826254826, 
    'black_finite': 0.0009652509652509653
    }
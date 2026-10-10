import pandas as pd

CARD_INDEX_PATH = "data/2020_panini_prizm_card_index.csv"

card_index = pd.read_csv(CARD_INDEX_PATH)


def get_card_market_factor(date):
    row = card_index[card_index["date"] == date]

    if row.empty:
        raise ValueError(f"No card market data for {date}")

    return float(row.iloc[0]["factor"])
import pandas as pd

CARD_INDEX_PATH = "data/2020_panini_prizm_card_index.csv"
card_index = pd.read_csv(CARD_INDEX_PATH)
card_index["date"] = pd.to_datetime(card_index["date"])
card_index = card_index.sort_values("date")

def get_card_market_factor(date):

    date = pd.to_datetime(date)
    available_data = card_index[
        card_index["date"] <= date
    ]

    if available_data.empty:
        raise ValueError(f"No card market data available for {date}")
    
    return float(available_data.iloc[-1]["factor"])
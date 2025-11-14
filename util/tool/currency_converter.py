import json, random


def currency_converter_to_huf(amount: float, currency: str):
    print(f"Árfolyam számítás indítása. Összeg: {amount}, pénznem: {currency}")

    converted_amount: float

    if currency.upper() == "HUF":
        converted_amount = amount
    else:
        converted_amount = amount * random.uniform(0.1, 5)

    print(f"Átváltott összeg: {converted_amount} HUF")

    return json.dumps({"amount": converted_amount, "currency": "HUF"})

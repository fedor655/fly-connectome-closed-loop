"""Разведка метаданных коннектома FlyWire v783.

Задача: понять, какие классы нейронов доступны, и найти корректные интерфейсы
мозг<->тело. FlyWire — коннектом ГОЛОВНОГО мозга (без вентрального нервного тяжа),
поэтому:
  - выход к ногам   = нисходящие нейроны (descending, DN)
  - вход от ног     = восходящие нейроны (ascending, AN)
Механосенсоров ног в этом датасете быть не должно — проверяем это явно.
"""
import os

import pandas as pd

DATA_DIR = "/home/fedor/fly-brain-data"
COMP = os.path.join(DATA_DIR, "2025_Completeness_783.csv")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv(COMP)
print("=" * 78)
print("2025_Completeness_783.csv")
print("=" * 78)
print("shape:", df.shape)
print("columns:", list(df.columns))
print()
print(df.head(5))
print()

for col in df.columns:
    nun = df[col].nunique(dropna=True)
    print(f"--- {col!r}  (уникальных: {nun}, пропусков: {df[col].isna().sum()})")
    if nun <= 60:
        vc = df[col].value_counts(dropna=False)
        print(vc.to_string())
    else:
        print(df[col].dropna().astype(str).head(8).to_string())
    print()

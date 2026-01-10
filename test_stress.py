import pandas as pd
import random

noms = ["Jean Dupont", "Marie Martin", "Pierre Bernard"] * 67
data = []
for i in range(1000):
    nom = random.choice(noms)
    data.append({
        'nom': nom,
        'email': f"{nom.lower().replace(' ', '.')}@test{random.randint(1,10)}.fr",
        'telephone': f"06{random.randint(10000000,99999999)}",
        'entreprise': f"Test{random.randint(1,20)}"
    })

df = pd.DataFrame(data)
df.to_csv("test_1000lignes.csv", index=False)
print("✅ test_1000lignes.csv créé (1000 lignes)")

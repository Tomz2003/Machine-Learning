import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "khach_vay.csv")
X = df.drop(columns="tra_dung_han"); y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
print("min_samples_leaf | CV accuracy")
diem = {}
for n in [1,2,5,10,20,40]:
    m = DecisionTreeClassifier(min_samples_leaf=n, random_state=0)
    diem[n] = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    print(f"{n:16d} | {diem[n]:.4f}")
tot_nhat = max(diem, key=diem.get)
m = DecisionTreeClassifier(min_samples_leaf=tot_nhat, random_state=0).fit(X_tr,y_tr)
base = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr,y_tr)
print(f"Gia tri tot nhat: {tot_nhat}")
print(f"Cay cat tia: depth={m.get_depth()}, leaves={m.get_n_leaves()}, test accuracy={m.score(X_te,y_te):.4f}")
print(f"Cay sau 5: test accuracy={base.score(X_te,y_te):.4f}")
print("Nhan xet: min_samples_leaf buộc mỗi lá phải có đủ số khách, giúp cây tránh chia quá nhỏ; so sánh accuracy test để xem cách này có tốt hơn cây sâu 5 hay không.")

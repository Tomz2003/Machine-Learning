import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "khach_vay.csv")
X = df.drop(columns="tra_dung_han"); y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
rng = np.random.default_rng(0)
X_tr = X_tr.copy(); X_te = X_te.copy()
X_tr["so_ngau_nhien"] = rng.random(len(X_tr))
X_te["so_ngau_nhien"] = rng.random(len(X_te))
rung = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr,y_tr)
print(f"Accuracy test: {rung.score(X_te,y_te):.4f}")
print("Feature_importances_ (giam impurity):")
for cot, imp in sorted(zip(X_tr.columns, rung.feature_importances_), key=lambda z:z[1], reverse=True): print(f"  {cot:22s} {imp:.4f}")
perm = permutation_importance(rung, X_te, y_te, n_repeats=20, random_state=0)
rank = sorted(zip(X_te.columns, perm.importances_mean, perm.importances_std), key=lambda z:z[1], reverse=True)
print("Permutation importance tren tap kiem tra (mean +/- std):")
for i,(cot, imp, std) in enumerate(rank,1): print(f"  {i}. {cot:22s} {imp:.4f} +/- {std:.4f}")
print(f"Cot so_ngau_nhien xep hang {1 + [x[0] for x in rank].index('so_ngau_nhien')}/{len(rank)} theo permutation importance.")
print("Nhan xet: feature_importances_ cho biết cột được dùng để chia trong rừng và có thể ưu ái cột nhiều giá trị khác nhau. Permutation importance đo accuracy giảm bao nhiêu khi xáo trộn cột trên tập kiểm tra; cột ngẫu nhiên thường gần 0 vì không mang thông tin thật.")

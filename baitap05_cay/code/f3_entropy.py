import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "khach_vay.csv")
X = df.drop(columns="tra_dung_han"); y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
print("Do sau | accuracy hoc | diem CV")
diem = {}
for d in range(1, 16):
    m = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0)
    diem[d] = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    m.fit(X_tr, y_tr)
    print(f"{d:7d} | {m.score(X_tr,y_tr):.4f} | {diem[d]:.4f}")
d_tot = max(diem, key=diem.get)
m = DecisionTreeClassifier(criterion="entropy", max_depth=d_tot, random_state=0).fit(X_tr, y_tr)
print(f"Do sau tot nhat entropy: {d_tot}")
print(f"Accuracy test entropy: {accuracy_score(y_te,m.predict(X_te)):.4f}")
base = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr,y_tr)
print(f"So sanh cay Gini sau 5: accuracy test={base.score(X_te,y_te):.4f}")
print("Nhan xet: Gini va entropy co the chon cach chia/ do sau khac nhau, nhung neu accuracy test gan nhau thi doi thước đo khong lam ket qua thay doi nhieu.")

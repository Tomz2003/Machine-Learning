# Bai 4: KNN bo phieu co trong so theo khoang cach
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_hoc, X_kt, y_hoc, y_kt = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
print("So sanh KNN bo phieu thuong va bo phieu co trong so")
for k in [1, 13, 101]:
    for weights in ["uniform", "distance"]:
        mo_hinh = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X_hoc, y_hoc)
        acc = accuracy_score(y_kt, mo_hinh.predict(X_kt))
        print(f"k={k:3d}, weights={weights:8s}, accuracy={acc:.4f}")
print("Nhan xet: Trong so distance tang anh huong cua cac diem gan mau moi, nhung voi k rat lon van co the lam mo hinh qua don gian.")

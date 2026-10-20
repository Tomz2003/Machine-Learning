# Bai 2: So sanh khoang cach Manhattan va Euclid
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_hoc, X_kt, y_hoc, y_kt = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
ket_qua = {}
for metric in ["euclidean", "manhattan"]:
    diem_cv = {}
    for k in range(1, 52, 2):
        diem_cv[k] = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric=metric),
            X_hoc, y_hoc, cv=5
        ).mean()
    k_tot = max(diem_cv, key=diem_cv.get)
    mo_hinh = KNeighborsClassifier(n_neighbors=k_tot, metric=metric).fit(X_hoc, y_hoc)
    acc = accuracy_score(y_kt, mo_hinh.predict(X_kt))
    ket_qua[metric] = (k_tot, acc, diem_cv[k_tot])
    print(f"{metric}: k tot nhat = {k_tot}, CV = {diem_cv[k_tot]:.4f}, accuracy test = {acc:.4f}")
print("Nhan xet: ", end="")
if abs(ket_qua["euclidean"][1] - ket_qua["manhattan"][1]) < 0.05:
    print("Doi sang Manhattan khong lam accuracy thay doi nhieu tren tap kiem tra nay.")
else:
    print("Doi sang Manhattan lam accuracy thay doi ro hon tren tap kiem tra nay.")

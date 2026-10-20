# Bai 1: Tu cai dat KNN, khong dung thu vien KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    # Tinh khoang cach Euclid tu mau moi den tat ca mau hoc
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    # Lay chi so cua k mau gan nhat
    chi_so_gan = np.argsort(khoang_cach)[:k]
    nhan_lang_gieng = y_hoc[chi_so_gan]
    # Chon nhan co nhieu phieu nhat
    nhan, so_phieu = np.unique(nhan_lang_gieng, return_counts=True)
    return nhan[np.argmax(so_phieu)]

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["nhan"].to_numpy()
X_hoc, X_kt, y_hoc, y_kt = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
du_doan_tu_viet = np.array([knn_du_doan(X_hoc, y_hoc, x, 13) for x in X_kt])
acc_tu_viet = accuracy_score(y_kt, du_doan_tu_viet)
mo_hinh = KNeighborsClassifier(n_neighbors=13).fit(X_hoc, y_hoc)
du_doan_thu_vien = mo_hinh.predict(X_kt)
acc_thu_vien = accuracy_score(y_kt, du_doan_thu_vien)
print(f"Accuracy KNN tu viet: {acc_tu_viet:.4f}")
print(f"Accuracy KNeighborsClassifier: {acc_thu_vien:.4f}")
print(f"So du doan khac nhau: {np.sum(du_doan_tu_viet != du_doan_thu_vien)}")

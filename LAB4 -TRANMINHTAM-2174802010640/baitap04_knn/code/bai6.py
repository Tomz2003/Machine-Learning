# Bai 6: Do thoi gian fit va predict cua KNN
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
X_moi = rng.random((1000, 2))
print(f"{'So mau hoc':>12} {'Fit (giay)':>14} {'Predict 1000 mau (giay)':>26}")
for n in [1000, 10000, 100000]:
    X_hoc = rng.random((n, 2))
    y_hoc = rng.integers(0, 2, size=n)
    mo_hinh = KNeighborsClassifier(n_neighbors=13)
    bat_dau = time.perf_counter()
    mo_hinh.fit(X_hoc, y_hoc)
    thoi_gian_fit = time.perf_counter() - bat_dau
    bat_dau = time.perf_counter()
    mo_hinh.predict(X_moi)
    thoi_gian_predict = time.perf_counter() - bat_dau
    print(f"{n:12d} {thoi_gian_fit:14.6f} {thoi_gian_predict:26.6f}")
print("Nhan xet: KNN hoc luoi vi buoc fit chu yeu chi luu du lieu, nen nhanh; khi predict phai tinh khoang cach den nhieu mau da luu, nen du doan cham dan khi du lieu tang.")

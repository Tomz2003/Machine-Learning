# Bai 5: Chon k cho KNN hoi quy bang kiem dinh cheo
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_hoc, X_kt, y_hoc, y_kt = train_test_split(
    X, y, test_size=0.2, random_state=42
)
diem_cv = {}
for k in range(1, 21):
    diem_am = cross_val_score(
        KNeighborsRegressor(n_neighbors=k), X_hoc, y_hoc,
        cv=5, scoring="neg_mean_absolute_error"
    )
    diem_cv[k] = -diem_am.mean()
k_tot = min(diem_cv, key=diem_cv.get)
knn = KNeighborsRegressor(n_neighbors=k_tot).fit(X_hoc, y_hoc)
duong_thang = LinearRegression().fit(X_hoc, y_hoc)
mae_knn = mean_absolute_error(y_kt, knn.predict(X_kt))
mae_duong = mean_absolute_error(y_kt, duong_thang.predict(X_kt))
print(f"k tot nhat theo CV: {k_tot}")
print(f"MAE CV nho nhat: {diem_cv[k_tot]:.4f}")
print(f"MAE KNN tren tap kiem tra: {mae_knn:.4f}")
print(f"MAE duong thang tren tap kiem tra: {mae_duong:.4f}")
print("Nhan xet: " + ("KNN co MAE thap hon duong thang tren tap kiem tra." if mae_knn < mae_duong else "KNN chua vuot duoc duong thang vi MAE tren tap kiem tra cao hon."))

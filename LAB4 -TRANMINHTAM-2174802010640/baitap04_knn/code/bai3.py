# Bai 3: Doi don vi chi tieu khi KNN khong chuan hoa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_mo_rong.csv")
df["chi_tieu_trieu"] = df["chi_tieu"] / 1000
y = df["qua_mon"]
cot_goc = ["gio_on", "diem_giua_ky", "chi_tieu"]
cot_moi = ["gio_on", "diem_giua_ky", "chi_tieu_trieu"]
X_hoc, X_kt, y_hoc, y_kt = train_test_split(
    df[cot_goc], y, test_size=0.25, random_state=17, stratify=y
)
# Doi don vi tren cot chi tieu cua ca hai tap, giu nguyen cach chia mau
X_hoc_moi = X_hoc.copy()
X_kt_moi = X_kt.copy()
X_hoc_moi["chi_tieu_trieu"] = X_hoc_moi["chi_tieu"] / 1000
X_kt_moi["chi_tieu_trieu"] = X_kt_moi["chi_tieu"] / 1000
X_hoc_moi = X_hoc_moi[cot_moi]
X_kt_moi = X_kt_moi[cot_moi]
acc_goc = accuracy_score(y_kt, KNeighborsClassifier(n_neighbors=7).fit(X_hoc, y_hoc).predict(X_kt))
acc_doi_don_vi = accuracy_score(y_kt, KNeighborsClassifier(n_neighbors=7).fit(X_hoc_moi, y_hoc).predict(X_kt_moi))
# Kiem tra them khi chuan hoa bang pipeline, cac cot giong nhau chi khac don vi
acc_chuan_goc = accuracy_score(y_kt, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)).fit(X_hoc, y_hoc).predict(X_kt))
acc_chuan_moi = accuracy_score(y_kt, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)).fit(X_hoc_moi, y_hoc).predict(X_kt_moi))
print(f"Ba cot, khong chuan hoa, don vi goc: {acc_goc:.4f}")
print(f"Ba cot, khong chuan hoa, chi_tieu tinh bang trieu: {acc_doi_don_vi:.4f}")
print(f"Ba cot, co chuan hoa, don vi goc: {acc_chuan_goc:.4f}")
print(f"Ba cot, co chuan hoa, chi_tieu tinh bang trieu: {acc_chuan_moi:.4f}")
print("Nhan xet: Khong chuan hoa, cot chi_tieu co thang do thay doi nen anh huong den khoang cach va lang gieng; sau chuan hoa, doi don vi khong lam thay doi ket qua dang ke.")

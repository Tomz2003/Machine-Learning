import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def gini(y):
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

def tim_cau_hoi(X, y):
    y = np.asarray(y)
    goc = gini(y)
    tot_nhat = (None, None, -1.0)
    for cot in X.columns:
        vals = np.sort(X[cot].unique())
        nguong_list = (vals[:-1] + vals[1:]) / 2
        for nguong in nguong_list:
            trai = X[cot].to_numpy() <= nguong
            if not trai.any() or trai.all(): continue
            g_sau = trai.mean()*gini(y[trai]) + (~trai).mean()*gini(y[~trai])
            giam = goc - g_sau
            if giam > tot_nhat[2]: tot_nhat = (cot, float(nguong), float(giam))
    return tot_nhat

df = pd.read_csv(DATA / "khach_vay.csv")
X = df.drop(columns="tra_dung_han"); y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
cot, nguong, giam = tim_cau_hoi(X_tr, y_tr)
print(f"So khach tap hoc: {len(X_tr)}")
print(f"Cau hoi tot nhat: {cot} <= {nguong:.2f}")
print(f"Muc giam Gini: {giam:.6f}")
print("Nhan xet: Cây thử các ngưỡng nằm giữa những giá trị liên tiếp của từng cột, tính Gini sau khi chia và chọn phép chia làm Gini giảm nhiều nhất. Với tập này, câu hỏi gốc là thu_nhap <= 15.05.")

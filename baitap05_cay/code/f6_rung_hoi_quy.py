import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "gia_nha.csv")
X = df[["dien_tich"]]; y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
cay = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr,y_tr)
duong = LinearRegression().fit(X_tr,y_tr)
rung = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr,y_tr)
print("MAE tren tap kiem tra (ty dong):")
for ten,m in [("Cay sau 3",cay),("Duong thang",duong),("Random Forest 200 cay",rung)]: print(f"  {ten:24s}: {mean_absolute_error(y_te,m.predict(X_te)):.4f}")
can = pd.DataFrame({"dien_tich":[250.0]})
print(f"Du doan gia nha 250 m2 (ty dong): cay={cay.predict(can)[0]:.2f}, duong thang={duong.predict(can)[0]:.2f}, rung={rung.predict(can)[0]:.2f}")
print("Nhan xet: Rừng cây quyết định không ngoại suy tốt ra ngoài khoảng diện tích đã học. Mỗi cây trả về giá trị trung bình ở lá mà căn nhà rơi vào, nên dự đoán của rừng là trung bình các dự đoán đó và thường nằm trong khoảng giá đã thấy ở tập học.")

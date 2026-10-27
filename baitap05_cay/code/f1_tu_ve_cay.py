import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
for depth, filename in [(None, "cay_nam_la.png"), (2, "cay_sau_2.png")]:
    model = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0) if depth is None else DecisionTreeClassifier(max_depth=2, random_state=0)
    model.fit(X_tr, y_tr)
    plt.figure(figsize=(15, 8))
    plot_tree(model, feature_names=X.columns, class_names=["tre han", "dung han"], filled=True, rounded=True)
    plt.tight_layout()
    plt.savefig(ROOT / filename, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"{filename}: do sau={model.get_depth()}, so la={model.get_n_leaves()}, accuracy test={model.score(X_te, y_te):.4f}")
print("Nhan xet: Cay sau 2 chi co hai nhanh con o tang thu hai; neu hai la cung ket luan thi viec tach tiep khong doi nhan du doan, nhung co the tach de lam cac nhanh thuần hon theo Gini/entropy.")

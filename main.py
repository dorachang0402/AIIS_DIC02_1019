"""
50_Startups 複迴歸 (Multiple Linear Regression) 分析
遵循 CRISP-DM (跨產業數據探勘標準流程) 6 大步驟
"""

import sys
import io

# 設定標準輸出為 UTF-8，避免 Windows 繁體中文終端機 (cp950) 編碼錯誤
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 設定字型支援 (包含微軟正黑體、簡黑、Segoe UI 與 Unicode 負號)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# 1. Business Understanding (商業理解)
# ==============================================================================
# 【目標】：
# 建立複迴歸 (Multiple Linear Regression) 模型，探討創業公司的研發支出 (R&D Spend)、
# 行政支出 (Administration)、營銷支出 (Marketing Spend) 及所在州別 (State) 與最終利潤 (Profit) 的關聯。
#
# 【商業價值】：
# 1. 創投評估 (VC Investment Evaluation)：幫助投資機構在審查潛力新創時，以量化數據評估預期獲利能力。
# 2. 資源分配 (Budget Allocation)：協助創業者釐清各項資本支出（特別是 R&D vs. Marketing）對利潤的邊際貢獻，制定最佳預算策略。
# 3. 風險控管 (Risk Management)：識別過高行政開銷是否稀釋獲利，避免無效益資本浪費。
print("=" * 60)
print("1. Business Understanding: 目標與商業價值確立完成")
print("=" * 60)

# ==============================================================================
# 2. Data Understanding (資料理解)
# ==============================================================================
url = "https://raw.githubusercontent.com/ShreyaPatil1199/50_Startups_Profit/main/50_Startups.csv"
df = pd.read_csv(url)

print("\n" + "=" * 60)
print("2. Data Understanding (資料理解)")
print("=" * 60)
print("\n--- [2.1] 資料前 5 筆 ---")
print(df.head())

print("\n--- [2.2] 資料型態與非空值統計 ---")
df.info()

print("\n--- [2.3] 缺失值檢查 ---")
print(df.isnull().sum())

print("\n--- [2.4] 數值型特徵敘述性統計 (Describe) ---")
print(df.describe())

print("\n--- [2.5] 類別變數 (State) 分佈 ---")
print(df["State"].value_counts())

# ==============================================================================
# 3. Data Preparation (資料準備)
# ==============================================================================
print("\n" + "=" * 60)
print("3. Data Preparation (資料準備)")
print("=" * 60)

# (1) 分離自變數 (X) 與因變數 (y)
X = df.iloc[:, :-1]  # 前 4 欄：R&D Spend, Administration, Marketing Spend, State
y = df.iloc[:, -1]   # 第 5 欄：Profit

# (2) 類別變數 One-Hot Encoding
# 使用 drop='first' 移除第一個類別（基準為 California），以避免「啞變數陷阱 (Dummy Variable Trap)」所導致的完全共線性
ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(drop="first"), ["State"])
    ],
    remainder="passthrough"  # 其他數值欄位保持原樣
)

X_encoded = ct.fit_transform(X)

# (3) 拆分訓練集與測試集 (80% 訓練, 20% 測試)
# 使用 random_state=0 以符合特徵選取評估之實驗設定
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)

print(f"訓練集維度: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"測試集維度: X_test  = {X_test.shape}, y_test  = {y_test.shape}")

# ==============================================================================
# 4. Modeling (模型建立) & Feature Selection (特徵選取評估)
# ==============================================================================
print("\n" + "=" * 60)
print("4. Modeling & Feature Selection (特徵選取比較評估)")
print("=" * 60)

# (1) 基準全特徵模型擬合
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("已成功使用 scikit-learn LinearRegression 擬合訓練集數據！")

# (2) 特徵數量消長實驗 (Feature Selection Comparison)
# 準備 dummy 編碼供特徵子集評估
df_encoded = pd.get_dummies(df, columns=["State"], dtype=float)
X_features = df_encoded.drop(columns=["Profit"])
y_features = df_encoded["Profit"]
X_tr, X_te, y_tr, y_te = train_test_split(X_features, y_features, test_size=0.2, random_state=0)

features_display = [
    ["R&D Spend"],
    ["R&D Spend", "Marketing Spend"],
    ["R&D Spend", "Marketing Spend", "State_New York"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "State_California"]
]

features_fit = [
    ["R&D Spend"],
    ["R&D Spend", "Marketing Spend"],
    ["R&D Spend", "Marketing Spend", "State_New York"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"]
]

feature_results = []
for num_feat, (feat_disp, feat_fit) in enumerate(zip(features_display, features_fit), 1):
    m = LinearRegression()
    m.fit(X_tr[feat_fit], y_tr)
    preds = m.predict(X_te[feat_fit])
    
    val_rmse = np.sqrt(mean_squared_error(y_te, preds))
    val_r2 = r2_score(y_te, preds)
    
    feature_results.append({
        "Number of Features": num_feat,
        "Selected Features": f"[{', '.join(feat_disp)}]",
        "RMSE": val_rmse,
        "R-squared": val_r2
    })

feature_results_df = pd.DataFrame(feature_results)

print("\n--- 特徵選取評估結果 (Feature Selection Summary Table) ---")
# 設定 pandas 顯示寬度避免被截斷
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(feature_results_df.to_string(index=False))

# (3) 繪製特徵數量與 RMSE、R-squared 之關係圖 (專業視覺化設計，兼具學術與報表質感)
fig_feat, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

x_nums = feature_results_df["Number of Features"].values
rmse_vals = feature_results_df["RMSE"].values
r2_vals = feature_results_df["R-squared"].values

# ----------------- 子圖 1: RMSE by Number of Features -----------------
rmse_color = "#4f46e5"  # 沉穩靛藍色 (Indigo)
ax_rmse.plot(x_nums, rmse_vals, color=rmse_color, linewidth=2.5, marker="o", 
             markersize=8, markerfacecolor="#ffffff", markeredgecolor=rmse_color, markeredgewidth=2.2, label="RMSE")
ax_rmse.fill_between(x_nums, rmse_vals, min(rmse_vals) * 0.98, color=rmse_color, alpha=0.08)

# 標註各點數值與突顯最佳點 (k=2)
best_rmse_idx = int(np.argmin(rmse_vals))
for idx, (x, y_val) in enumerate(zip(x_nums, rmse_vals)):
    if idx == best_rmse_idx:
        ax_rmse.plot(x, y_val, marker="o", markersize=11, markerfacecolor="#e11d48", markeredgecolor="#ffffff", markeredgewidth=2)
        ax_rmse.annotate(f"★ Optimal (k={x})\nRMSE: {y_val:.2f}",
                         xy=(x, y_val), xytext=(x, y_val - 130),
                         ha="center", fontsize=9.5, fontweight="bold", color="#991b1b",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe4e6", edgecolor="#f43f5e", alpha=0.9),
                         arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.5))
    else:
        ax_rmse.text(x, y_val + 25, f"{y_val:.1f}", ha="center", va="bottom", fontsize=8.5, color="#475569")

ax_rmse.set_title("RMSE by Number of Features", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
ax_rmse.set_xlabel("Number of Features", fontsize=11, labelpad=8, color="#334155")
ax_rmse.set_ylabel("RMSE (Lower is Better)", fontsize=11, labelpad=8, color="#334155")
ax_rmse.set_xticks(x_nums)
ax_rmse.set_ylim(min(rmse_vals) - 180, max(rmse_vals) + 120)

# ----------------- 子圖 2: R-squared by Number of Features -----------------
r2_color = "#059669"  # 翡翠綠 (Emerald)
ax_r2.plot(x_nums, r2_vals, color=r2_color, linewidth=2.5, marker="o", 
           markersize=8, markerfacecolor="#ffffff", markeredgecolor=r2_color, markeredgewidth=2.2, label="R-squared")
ax_r2.fill_between(x_nums, r2_vals, min(r2_vals) * 0.998, color=r2_color, alpha=0.08)

# 標註各點數值與突顯最佳點 (k=2)
best_r2_idx = int(np.argmax(r2_vals))
for idx, (x, y_val) in enumerate(zip(x_nums, r2_vals)):
    if idx == best_r2_idx:
        ax_r2.plot(x, y_val, marker="o", markersize=11, markerfacecolor="#059669", markeredgecolor="#ffffff", markeredgewidth=2)
        ax_r2.annotate(f"★ Optimal (k={x})\nR²: {y_val:.6f}",
                       xy=(x, y_val), xytext=(x, y_val + 0.0022),
                       ha="center", fontsize=9.5, fontweight="bold", color="#065f46",
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#10b981", alpha=0.9),
                       arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5))
    else:
        ax_r2.text(x, y_val - 0.0012, f"{y_val:.4f}", ha="center", va="top", fontsize=8.5, color="#475569")

ax_r2.set_title("R-squared by Number of Features", fontsize=13, fontweight="bold", pad=12, color="#1e293b")
ax_r2.set_xlabel("Number of Features", fontsize=11, labelpad=8, color="#334155")
ax_r2.set_ylabel("R-squared (Higher is Better)", fontsize=11, labelpad=8, color="#334155")
ax_r2.set_xticks(x_nums)
ax_r2.set_ylim(min(r2_vals) - 0.002, max(r2_vals) + 0.004)

# 美化兩圖的外框與網格
for ax in (ax_rmse, ax_r2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9.5)
    ax.grid(True, linestyle="--", linewidth=0.7, color="#e2e8f0", alpha=0.9)

plt.tight_layout()
feat_chart_path = "feature_selection_plots.png"
fig_feat.savefig(feat_chart_path, dpi=300, bbox_inches="tight")
plt.close(fig_feat)
print(f"\n特徵選取趨勢圖已儲存至: {feat_chart_path}")

# (4) 將「Number of Features」特徵選取結果表格生成為現代報告風圖表
fig_tbl, ax_tbl = plt.subplots(figsize=(15, 3.8), dpi=300)
ax_tbl.axis("off")

table_data = [
    [
        f"★ {row['Number of Features']} (Optimal)" if row["Number of Features"] == 2 else f"{row['Number of Features']}",
        row["Selected Features"],
        f"{row['RMSE']:.6f}",
        f"{row['R-squared']:.6f}"
    ]
    for _, row in feature_results_df.iterrows()
]
col_headers = ["Number of Features", "Selected Features", "RMSE", "R-squared"]
col_widths = [0.20, 0.52, 0.14, 0.14]

table_obj = ax_tbl.table(
    cellText=table_data,
    colLabels=col_headers,
    colWidths=col_widths,
    cellLoc="left",
    loc="center",
    bbox=[0.01, 0.06, 0.98, 0.88]
)
table_obj.auto_set_font_size(False)
table_obj.set_fontsize(11)

# 為表格套用高階現代配色 (Slate Header + 斑馬紋 + 最佳行綠色高亮)
for (row_idx, col_idx), cell in table_obj.get_celld().items():
    ha_align = "center" if col_idx == 0 else ("right" if col_idx in [2, 3] else "left")
    if row_idx == 0:
        # 表頭：深色沉穩石板藍
        cell.set_facecolor("#1e293b")
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.2)
        cell.set_text_props(color="#ffffff", weight="bold", ha=ha_align)
    elif row_idx == 2:
        # 最佳模型列 (k=2)：淺翡翠綠背景突顯
        cell.set_facecolor("#ecfdf5")
        cell.set_edgecolor("#a7f3d0")
        cell.set_linewidth(1.0)
        cell.set_text_props(color="#065f46", weight="bold" if col_idx in [0, 2, 3] else "normal", ha=ha_align)
    else:
        # 其他列：白與淡灰交替斑馬紋
        row_bg = "#f8fafc" if row_idx % 2 == 1 else "#ffffff"
        cell.set_facecolor(row_bg)
        cell.set_edgecolor("#e2e8f0")
        cell.set_linewidth(0.8)
        cell.set_text_props(color="#334155", ha=ha_align)

plt.tight_layout()
table_chart_path = "number_of_features_table.png"
fig_tbl.savefig(table_chart_path, dpi=300, bbox_inches="tight")
plt.close(fig_tbl)
print(f"Number of Features 表格圖表已儲存至: {table_chart_path}")

# (5) 額外生成「整合式特徵選取綜合看板圖 (Dashboard)」便於報告完整呈現
fig_dash = plt.figure(figsize=(15, 9.8), dpi=300)
gs = fig_dash.add_gridspec(2, 2, height_ratios=[1.2, 0.95], hspace=0.35, wspace=0.18)

ax_d_rmse = fig_dash.add_subplot(gs[0, 0])
ax_d_r2 = fig_dash.add_subplot(gs[0, 1])
ax_d_tbl = fig_dash.add_subplot(gs[1, :])

# 重現折線圖至 Dashboard
ax_d_rmse.plot(x_nums, rmse_vals, color="#4f46e5", linewidth=2.5, marker="o",
               markersize=7.5, markerfacecolor="#ffffff", markeredgecolor="#4f46e5", markeredgewidth=2)
ax_d_rmse.fill_between(x_nums, rmse_vals, min(rmse_vals) * 0.98, color="#4f46e5", alpha=0.08)
ax_d_rmse.plot(2, rmse_vals[1], marker="o", markersize=10, markerfacecolor="#e11d48", markeredgecolor="#ffffff", markeredgewidth=2)
ax_d_rmse.annotate(f"Optimal (k=2)\nRMSE: {rmse_vals[1]:.2f}",
                   xy=(2, rmse_vals[1]), xytext=(2, rmse_vals[1] - 130),
                   ha="center", fontsize=9, fontweight="bold", color="#991b1b",
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffe4e6", edgecolor="#f43f5e", alpha=0.9),
                   arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.3))
for idx, (x, y_val) in enumerate(zip(x_nums, rmse_vals)):
    if idx != 1:
        ax_d_rmse.text(x, y_val + 25, f"{y_val:.1f}", ha="center", va="bottom", fontsize=8, color="#475569")
ax_d_rmse.set_title("RMSE by Number of Features", fontsize=12, fontweight="bold", pad=10, color="#1e293b")
ax_d_rmse.set_xlabel("Number of Features", fontsize=10, color="#334155")
ax_d_rmse.set_ylabel("RMSE", fontsize=10, color="#334155")
ax_d_rmse.set_xticks(x_nums)
ax_d_rmse.set_ylim(min(rmse_vals) - 180, max(rmse_vals) + 120)

ax_d_r2.plot(x_nums, r2_vals, color="#059669", linewidth=2.5, marker="o",
             markersize=7.5, markerfacecolor="#ffffff", markeredgecolor="#059669", markeredgewidth=2)
ax_d_r2.fill_between(x_nums, r2_vals, min(r2_vals) * 0.998, color="#059669", alpha=0.08)
ax_d_r2.plot(2, r2_vals[1], marker="o", markersize=10, markerfacecolor="#059669", markeredgecolor="#ffffff", markeredgewidth=2)
ax_d_r2.annotate(f"Optimal (k=2)\nR²: {r2_vals[1]:.6f}",
                 xy=(2, r2_vals[1]), xytext=(2, r2_vals[1] + 0.0022),
                 ha="center", fontsize=9, fontweight="bold", color="#065f46",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecfdf5", edgecolor="#10b981", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3))
for idx, (x, y_val) in enumerate(zip(x_nums, r2_vals)):
    if idx != 1:
        ax_d_r2.text(x, y_val - 0.0012, f"{y_val:.4f}", ha="center", va="top", fontsize=8, color="#475569")
ax_d_r2.set_title("R-squared by Number of Features", fontsize=12, fontweight="bold", pad=10, color="#1e293b")
ax_d_r2.set_xlabel("Number of Features", fontsize=10, color="#334155")
ax_d_r2.set_ylabel("R-squared", fontsize=10, color="#334155")
ax_d_r2.set_xticks(x_nums)
ax_d_r2.set_ylim(min(r2_vals) - 0.002, max(r2_vals) + 0.004)

for ax in (ax_d_rmse, ax_d_r2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.7, color="#e2e8f0", alpha=0.9)

# 表格重現至 Dashboard
ax_d_tbl.axis("off")
d_table_obj = ax_d_tbl.table(
    cellText=table_data,
    colLabels=col_headers,
    colWidths=col_widths,
    cellLoc="left",
    loc="center",
    bbox=[0.01, 0.05, 0.98, 0.88]
)
d_table_obj.auto_set_font_size(False)
d_table_obj.set_fontsize(10.5)

for (row_idx, col_idx), cell in d_table_obj.get_celld().items():
    ha_align = "center" if col_idx == 0 else ("right" if col_idx in [2, 3] else "left")
    if row_idx == 0:
        cell.set_facecolor("#1e293b")
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.2)
        cell.set_text_props(color="#ffffff", weight="bold", ha=ha_align)
    elif row_idx == 2:
        cell.set_facecolor("#ecfdf5")
        cell.set_edgecolor("#a7f3d0")
        cell.set_linewidth(1.0)
        cell.set_text_props(color="#065f46", weight="bold" if col_idx in [0, 2, 3] else "normal", ha=ha_align)
    else:
        row_bg = "#f8fafc" if row_idx % 2 == 1 else "#ffffff"
        cell.set_facecolor(row_bg)
        cell.set_edgecolor("#e2e8f0")
        cell.set_linewidth(0.8)
        cell.set_text_props(color="#334155", ha=ha_align)

fig_dash.suptitle("Feature Selection Performance & Model Comparison (CRISP-DM Stage 4)", 
                  fontsize=15, fontweight="bold", y=0.98, color="#0f172a")

dashboard_chart_path = "feature_selection_dashboard.png"
fig_dash.savefig(dashboard_chart_path, dpi=300, bbox_inches="tight")
plt.close(fig_dash)
print(f"特徵選取綜合看板圖已儲存至: {dashboard_chart_path}")

# ==============================================================================
# 5. Evaluation (模型評估)
# ==============================================================================
print("\n" + "=" * 60)
print("5. Evaluation (全模型評估)")
print("=" * 60)

# (1) 測試集預測
y_pred = regressor.predict(X_test)

# (2) 計算評估指標
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- [5.1] 測試集效能評估指標 ---")
print(f"R-squared (R^2): {r2:.6f}  (解釋力達 {r2 * 100:.2f}%)")
print(f"MAE           : ${mae:,.2f}")
print(f"MSE           : {mse:,.2f}")
print(f"RMSE          : {rmse:.6f}")

# (3) 輸出截距與迴歸係數
feature_names = ct.get_feature_names_out()
print("\n--- [5.2] 迴歸方程式參數解析 ---")
print(f"截距項 (Intercept, β₀): ${regressor.intercept_:,.2f}")
for name, coef in zip(feature_names, regressor.coef_):
    clean_name = name.replace("remainder__", "").replace("encoder__", "")
    print(f"係數 - {clean_name:<20}: {coef:>10.4f}")

# (4) 視覺化評估 (實際值 vs. 預測值 & 殘差圖)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 圖 1: 實際值 vs 預測值
ax1.scatter(y_test, y_pred, color="#2563eb", edgecolors="k", s=80, alpha=0.8, label="Test Set Data")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
ax1.set_title("Actual vs. Predicted Profit (50 Startups)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Actual Profit ($)", fontsize=11)
ax1.set_ylabel("Predicted Profit ($)", fontsize=11)
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

# 圖 2: 殘差圖 (Residuals)
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, color="#059669", edgecolors="k", s=80, alpha=0.8)
ax2.axhline(0, color="#ef4444", linestyle="--", linewidth=2)
ax2.set_title("Residual Plot (殘差分佈)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Predicted Profit ($)", fontsize=11)
ax2.set_ylabel("Residuals ($) [Actual - Predicted]", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
chart_path = "model_evaluation_plots.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"\n全模型評估圖表已輸出儲存至: {chart_path}")

# ==============================================================================
# 6. Deployment (部署準備與預測示範)
# ==============================================================================
print("\n" + "=" * 60)
print("6. Deployment (部署準備與預測示範)")
print("=" * 60)


def predict_startup_profit(rd_spend: float, admin_spend: float, marketing_spend: float, state: str) -> float:
    """
    輸入新創業公司的支出與地區，利用訓練完成的 Pipeline 進行預測
    
    參數:
        rd_spend: 研發支出 (R&D Spend, 美元)
        admin_spend: 行政支出 (Administration, 美元)
        marketing_spend: 營銷支出 (Marketing Spend, 美元)
        state: 所在州別 ('California', 'Florida', 'New York')
        
    回傳:
        預估利潤 (Profit, 美元)
    """
    # 建立單筆資料 DataFrame，確保欄位名稱與型態一致
    new_data = pd.DataFrame({
        "R&D Spend": [rd_spend],
        "Administration": [admin_spend],
        "Marketing Spend": [marketing_spend],
        "State": [state]
    })
    
    # 透過先前 fit 好的 ColumnTransformer 轉碼
    new_data_encoded = ct.transform(new_data)
    
    # 模型預測
    predicted_profit = regressor.predict(new_data_encoded)[0]
    return predicted_profit


# 部署範例測試 1: 高研發新創
sample1_rd = 165000.0
sample1_admin = 136000.0
sample1_marketing = 470000.0
sample1_state = "New York"
pred1 = predict_startup_profit(sample1_rd, sample1_admin, sample1_marketing, sample1_state)

# 部署範例測試 2: 中型加州新創
sample2_rd = 100000.0
sample2_admin = 120000.0
sample2_marketing = 250000.0
sample2_state = "California"
pred2 = predict_startup_profit(sample2_rd, sample2_admin, sample2_marketing, sample2_state)

print(f"[範例 1] 州別: {sample1_state}, R&D: ${sample1_rd:,.0f}, Admin: ${sample1_admin:,.0f}, Marketing: ${sample1_marketing:,.0f}")
print(f"  ==> 預測 Profit: ${pred1:,.2f}")
print()
print(f"[範例 2] 州別: {sample2_state}, R&D: ${sample2_rd:,.0f}, Admin: ${sample2_admin:,.0f}, Marketing: ${sample2_marketing:,.0f}")
print(f"  ==> 預測 Profit: ${pred2:,.2f}")

import numpy as np
import pandas as pd
import statsmodels.api as sm

# 1. 讀取資料 (直接從 GitHub 讀取 raw 資料)
url = "https://raw.githubusercontent.com/yuhsin-Chen/yuhsin-chen.github.io/main/machine%20learning/Salary_Data.csv"
df = pd.read_csv(url)
x = df["YearsExperience"].values
y = df["Salary"].values
n = len(x)

# 2. 依照推導公式進行手算驗證（用來寫報告步驟）
x_bar = np.mean(x)
y_bar = np.mean(y)

Sxx = np.sum((x - x_bar) ** 2)
Syy = np.sum((y - y_bar) ** 2)  # 即 SST
Sxy = np.sum((x - x_bar) * (y - y_bar))

# 計算 OLS 估計值
beta_1 = Sxy / Sxx
beta_0 = y_bar - beta_1 * x_bar

# 計算預測值與殘差
y_hat = beta_0 + beta_1 * x
residuals = y - y_hat
SSE = np.sum(residuals**2)
SST = Syy
r_squared = 1 - (SSE / SST)

# 計算假設檢定統計量 (t 檢定)
df_e = n - 2
s_squared = SSE / df_e
se_beta_1 = np.sqrt(s_squared / Sxx)
t_stat = beta_1 / se_beta_1

print(f"=== 手算統計量結果 ===")
print(f"樣本數 n: {n}")
print(f"x 平均: {x_bar:.4f}, y 平均: {y_bar:.4f}")
print(f"Sxx: {Sxx:.4f}, Sxy: {Sxy:.4f}")
print(f"迴歸方程式: y = {beta_0:.4f} + {beta_1:.4f}x")
print(f"SST: {SST:.4f}, SSE: {SSE:.4f}")
print(f"判定係數 R^2: {r_squared:.4f}")
print(f"斜率標準誤 SE(beta_1): {se_beta_1:.4f}")
print(f"t 統計量: {t_stat:.4f}\n")

# 3. 呼叫 statsmodels 產出標準學術報表（可截圖貼在作業附錄）
X_with_const = sm.add_constant(df["YearsExperience"])
ols_model = sm.OLS(df["Salary"], X_with_const).fit()
print("=== 軟體標準分析摘要 ===")
print(ols_model.summary())

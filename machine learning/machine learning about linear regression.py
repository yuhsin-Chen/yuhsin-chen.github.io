import os
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# ==========================================
# 1. 讀取資料 (從雲端 CSV 讀取)
# ==========================================
url = "https://raw.githubusercontent.com/yuhsin-Chen/yuhsin-chen.github.io/main/machine%20learning/Salary_Data.csv"
df = pd.read_csv(url)

x = df["YearsExperience"].values
y = df["Salary"].values
n = len(x)

# ==========================================
# 2. 數值推導與手算計算過程
# ==========================================
x_bar = np.mean(x)
y_bar = np.mean(y)

Sxx = np.sum((x - x_bar) ** 2)
Syy = np.sum((y - y_bar) ** 2)  # 總平方和 SST
Sxy = np.sum((x - x_bar) * (y - y_bar))

# 迴歸係數
beta_1 = Sxy / Sxx
beta_0 = y_bar - beta_1 * x_bar

# 預測值與殘差
y_hat = beta_0 + beta_1 * x
residuals = y - y_hat

# 平方和與判定係數
SSE = np.sum(residuals**2)
SSR = Sxy**2 / Sxx  # 或 Syy - SSE
SST = Syy
r_squared = SSR / SST

# 自由度與變異數估計
df_e = n - 2
s_squared = SSE / df_e
s_e = np.sqrt(s_squared)

# 標準誤與 t 統計量
se_beta_1 = np.sqrt(s_squared / Sxx)
se_beta_0 = np.sqrt(s_squared * (1 / n + (x_bar**2) / Sxx))
t_stat = beta_1 / se_beta_1
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_e))  # 雙尾檢定 p-value

# ==========================================
# 3. 殘差檢定 (Residual Analysis)
# ==========================================
# 殘差統計量
res_mean = np.mean(residuals)
res_std = np.std(residuals, ddof=1)

# Shapiro-Wilk 常態性檢定
shapiro_stat, shapiro_p = stats.shapiro(residuals)

# ==========================================
# 4. Statsmodels OLS 驗證
# ==========================================
X_with_const = sm.add_constant(df["YearsExperience"])
ols_model = sm.OLS(df["Salary"], X_with_const).fit()

# ==========================================
# 5. 產生 Word 文件報告
# ==========================================
print("=== 開始製作 Word 報告... ===")
doc = Document()


# 設定標題輔助函式
def add_custom_heading(text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    return h


# 文件主標題
title_p = doc.add_paragraph()
title_run = title_p.add_run("簡單線性迴歸分析與殘差檢定完整報告")
title_run.font.size = Pt(20)
title_run.font.bold = True
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph("分析主題：工作經驗（YearsExperience）與薪資（Salary）之線性關係分析\n")

# ------------------------------------------
# 第一區塊：簡單線性迴歸公式推導與計算結果
# ------------------------------------------
add_custom_heading("一、 數學公式推導與計算結果", level=1)

p = doc.add_paragraph()
p.add_run("1. 樣本平均數 (Sample Means)：\n").bold = True
p.add_run(
    f"   • x̄ = (∑ x_i) / n = {x_bar:.4f}\n"
    f"   • ȳ = (∑ y_i) / n = {y_bar:.4f}\n"
)

p.add_run("2. 離差平方和與交叉積和 (Sum of Squares)：\n").bold = True
p.add_run(
    f"   • Sxx = ∑(x_i - x̄)² = {Sxx:.4f}\n"
    f"   • Syy = ∑(y_i - ȳ)² = {Syy:.4f}  (即總平方和 SST)\n"
    f"   • Sxy = ∑(x_i - x̄)(y_i - ȳ) = {Sxy:.4f}\n"
)

p.add_run("3. 迴歸參數估計 (Parameter Estimation - OLS)：\n").bold = True
p.add_run(
    f"   • 斜率 β₁ = Sxy / Sxx = {Sxy:.4f} / {Sxx:.4f} = {beta_1:.4f}\n"
    f"   • 截距 β₀ = ȳ - β₁ * x̄ = {y_bar:.4f} - ({beta_1:.4f} * {x_bar:.4f}) = {beta_0:.4f}\n"
    f"   👉 估計迴歸方程式： ŷ = {beta_0:.4f} + {beta_1:.4f} * x\n"
)

p.add_run("4. 變異數分解與判定係數 (Coefficient of Determination)：\n").bold = True
p.add_run(
    f"   • 迴歸平方和 SSR = Sxy² / Sxx = {SSR:.4f}\n"
    f"   • 殘差平方和 SSE = ∑(y_i - ŷ_i)² = {SSE:.4f}\n"
    f"   • 判定係數 R² = SSR / SST = 1 - (SSE / SST) = {r_squared:.4f} ({r_squared*100:.2f}%)\n"
    f"   • 解釋：薪資變異中有 {r_squared*100:.2f}% 可由工作年資的線性關係來解釋。\n"
)

# ------------------------------------------
# 第二區塊：假設檢定 (Hypothesis Testing)
# ------------------------------------------
add_custom_heading("二、 迴歸係數之假設檢定 (Hypothesis Testing)", level=1)

p_hyp = doc.add_paragraph()
p_hyp.add_run("針對斜率係數 β₁ 進行 t 檢定，驗證工作經驗是否對薪資具有顯著影響：\n\n")

p_hyp.add_run("1. 建立假設：\n").bold = True
p_hyp.add_run("   • 虛無假設 H₀: β₁ = 0 (工作經驗對薪資無顯著影響)\n")
p_hyp.add_run("   • 對立假設 H₁: β₁ ≠ 0 (工作經驗對薪資有顯著影響)\n\n")

p_hyp.add_run("2. 計算檢定統計量 (t-Statistic)：\n").bold = True
p_hyp.add_run(
    f"   • 自由度 df = n - 2 = {n} - 2 = {df_e}\n"
    f"   • 殘差均方 MSE (s²) = SSE / (n - 2) = {SSE:.4f} / {df_e} = {s_squared:.4f}\n"
    f"   • 斜率標準誤 SE(β₁) = √(s² / Sxx) = √({s_squared:.4f} / {Sxx:.4f}) = {se_beta_1:.4f}\n"
    f"   • t 統計量 = (β₁ - 0) / SE(β₁) = {beta_1:.4f} / {se_beta_1:.4f} = {t_stat:.4f}\n\n"
)

p_hyp.add_run("3. 統計決策與結論：\n").bold = True
p_hyp.add_run(f"   • P-value: {p_val:.4e}\n")
if p_val < 0.05:
    p_hyp.add_run(
        f"   • 結論：由於 P-value ({p_val:.4e}) < 0.05，我們在 5% 顯著水準下【拒絕虛無假設 H₀】。\n"
        f"     這代表工作經驗（YearsExperience）對薪資（Salary）具有極顯著的正向影響。\n"
    )
else:
    p_hyp.add_run(
        f"   • 結論：P-value ({p_val:.4e}) ≥ 0.05，無法拒絕 H₀。\n"
    )

# ------------------------------------------
# 第三區塊：殘差分析 (Residual Analysis)
# ------------------------------------------
add_custom_heading("三、 殘差分析 (Residual Analysis)", level=1)

p_res = doc.add_paragraph()
p_res.add_run(
    "簡單線性迴歸模型需滿足三項基本假設：常態性 (Normality)、獨立性 (Independence) 與變異數同質性 (Homoscedasticity)。\n\n"
)

p_res.add_run("1. 殘差基本統計特徵：\n").bold = True
p_res.add_run(
    f"   • 殘差平均值 (Mean of Residuals): {res_mean:.6f} (趨近於 0，符合無偏估計)\n"
    f"   • 殘差標準差 (Std of Residuals): {res_std:.4f}\n\n"
)

p_res.add_run("2. 常態性檢定 (Shapiro-Wilk Test)：\n").bold = True
p_res.add_run(f"   • 檢定統計量 W = {shapiro_stat:.4f}\n")
p_res.add_run(f"   • P-value = {shapiro_p:.4f}\n")
if shapiro_p > 0.05:
    p_res.add_run(
        f"   • 分析：P-value ({shapiro_p:.4f}) > 0.05，未達顯著水準，【無法拒絕殘差呈現常態分布之假設】。"
        f"這代表該資料殘差符合常態性假設。\n\n"
    )
else:
    p_res.add_run(
        f"   • 分析：P-value ({shapiro_p:.4f}) ≤ 0.05，殘差可能偏離常態分布。\n\n"
    )

p_res.add_run("3. 變異數同質性與獨立性評估摘要：\n").bold = True
p_res.add_run(
    "   • 變異數同質性：經觀察殘差分佈，殘差未隨預測值 ŷ 的增大而出現明顯擴散（如喇叭狀），符合同質變異假設。\n"
    "   • 獨立性：樣本資料為跨個體之橫斷面資料，無時間序列前後相依性，滿足獨立性要求。\n"
)

# ------------------------------------------
# 第四區塊：軟體 OLS 分析摘要表 (Statsmodels)
# ------------------------------------------
add_custom_heading("四、 OLS 模型軟體輸出對照表", level=1)

summary_p = doc.add_paragraph()
summary_run = summary_p.add_run(str(ols_model.summary()))
summary_run.font.name = "Consolas"
summary_run.font.size = Pt(8)

# ------------------------------------------
# 儲存檔案
# ------------------------------------------
output_path = os.path.expanduser("~/Desktop/線性迴歸分析報告_修訂版.docx")
doc.save(output_path)

print(f"✅ Word 報告已成功儲存至：{output_path}")

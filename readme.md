# power-forecasting

🔌 工廠用電預測系統（Power Forecasting System）

本專案旨在預測台灣工廠長期用電趨勢，採用 **XGBoost** 與 **Prophet** 雙模型架構，並整合時間特徵與週期性變化進行建模與預測。最終提供互動式查詢介面（Streamlit）以供內部使用者快速查看預測趨勢與模型績效比較。


## 📁 專案結構

```
power-forecasting/
│
├── data/                  # 原始資料與預測結果儲存
│   ├── raw/               # 原始 CSV 檔案
│   └── processed/         # 處理後資料（含特徵）
│
├── features/              # 特徵工程與未來特徵生成模組
│   ├── feature_generator.py
│   └── future_features.py
│
├── models/                # 模型訓練與預測邏輯
│   ├── xgboost_model.py
│   └── prophet_model.py
│
├── forecast/              # 預測結果生成與績效比較
│   ├── predictor.py
│   └── evaluator.py
│
├── app/                   # Streamlit 互動介面
│   └── dashboard.py
│
├── utils/                 # 公用工具（繪圖、資料處理）
│   └── visualization.py
│
├── main.py                # 執行流程總管
└── README.md              # 本說明文件
```

## 🔧 執行流程概述

1. **資料載入與處理**
   - 從 `data/raw/` 讀取歷史用電量資料
   - 時間欄位標準化、缺值處理

2. **特徵工程**
   - 產出時間相關特徵（年、月、週、日、是否週末）
   - 週期性特徵（sin/cos）以捕捉季節變化

3. **未來資料特徵生成**
   - 根據預測期間，自動生成未來特徵欄位，格式一致

4. **模型訓練與預測**
   - 使用 XGBoost 訓練含時間特徵的迴歸模型
   - 使用 Prophet 建立季節性與趨勢模型
   - 輸出預測結果至 `data/processed/`

5. **績效比較**
   - 計算 MAE、RMSE、MAPE
   - 可視化預測結果與誤差分布

6. **互動式應用介面（選用）**
   - 執行 `streamlit run app/dashboard.py`
   - 提供模型切換、預測時間調整、圖表展示與檔案下載功能

## 📦 套件依賴

- `pandas`
- `numpy`
- `xgboost`
- `prophet`
- `scikit-learn`
- `streamlit`
- `matplotlib` / `plotly`

安裝方式：

```bash
pip install -r requirements.txt
```

## 🚀 執行方式

### 1. 命令列模式

使用命令列模式執行預測：

```bash
python main.py --data data/raw/data.csv --periods 30 --freq D --plot
```

參數說明：
- `--data`: 輸入數據 CSV 檔案路徑（必要）
- `--output`: 預測結果輸出目錄（預設：`data/processed`）
- `--periods`: 預測期數（預設：30）
- `--freq`: 預測頻率（D=日，W=週，M=月，預設：D）
- `--plot`: 是否顯示視覺化圖表

### 2. 互動式儀表板

使用 Streamlit 互動式儀表板：

```bash
streamlit run app/dashboard.py
```

儀表板功能：
- 上傳歷史用電量數據
- 查看原始數據與趨勢圖
- 訓練模型
- 設定預測參數
- 查看預測結果
- 下載預測結果
- 季節性分析

## 📊 預測範圍與用途

- 適用於中長期工廠用電預測（如月/季/年）
- 可支援後續擴充：氣象資料、假日、工廠特殊排程等
- 初步專注於歷史用電 + 時間特徵（單變數預測）

## ✍️ 開發備註

- 不使用驗證集，直接比較模型在預測期間的績效
- 預測 horizon 可設定為未來 N 天、週或月
- 設計模組化架構，方便未來整合 API、自動化排程等

## 📬 聯絡人

開發者：`<你的名字>`
E-mail：`<你的 email>`  
版本：`v1.0`

## 🐳 Docker 開發環境

本專案提供 Docker 開發環境，方便在不同平台上進行開發和測試。

### 環境配置

專案包含以下 Docker 相關文件：

1. `Dockerfile` - 定義開發環境配置
2. `.dockerignore` - 指定不需要複製到容器中的文件
3. `docker-compose.yml` - 容器編排配置

### 使用方式

1. 構建 Docker 映像：
```bash
docker-compose build
```

2. 啟動開發環境：
```bash
docker-compose up -d
```

3. 進入容器：
```bash
docker-compose exec app bash
```

4. 關閉開發環境：
```bash
docker-compose down
```

5. 刪除所有標記為 <none> 的 images
```bash
docker image prune -f
```

### 開發環境特點

- 基於 Ubuntu 22.04
- 預裝 Python 3 和必要的開發工具
- 自動安裝專案依賴
- 支援即時代碼修改（通過卷掛載）
- 配置了適當的環境變量
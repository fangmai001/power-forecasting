# power-forecasting

## 🔌 工廠用電預測系統

本專案旨在預測台灣工廠長期用電趨勢，採用 **XGBoost** 與 **Prophet** 雙模型架構，整合時間特徵與週期性變化進行建模與預測。提供互動式查詢介面（Streamlit）供內部使用者快速查看預測趨勢與模型績效比較。

## 🏗️ 系統架構

### 專案結構

```
power-forecasting/
├── data/                  # 資料儲存
│   ├── raw/              # 原始資料
│   └── processed/        # 處理後資料
├── features/             # 特徵工程
│   ├── feature_generator.py
│   └── future_features.py
├── models/               # 模型模組
│   ├── xgboost_model.py
│   └── prophet_model.py
├── forecast/             # 預測模組
│   ├── predictor.py
│   └── evaluator.py
├── app/                  # 應用介面
│   └── dashboard.py
├── utils/                # 工具函式
│   └── visualization.py
├── main.py               # 主程式
└── README.md             # 說明文件
```

### 核心功能

1. **資料處理**
   - 歷史用電量資料載入與標準化
   - 時間欄位處理與缺值填補

2. **特徵工程**
   - 時間特徵（年、月、週、日、週末標記）
   - 週期性特徵（sin/cos 轉換）

3. **模型架構**
   - XGBoost：時間特徵迴歸模型
   - Prophet：季節性與趨勢模型

4. **預測分析**
   - 多期預測（日/週/月）
   - 績效評估（MAE、RMSE、MAPE）
   - 視覺化分析

## 🚀 快速開始

### 環境設置

```bash
# 安裝依賴套件
pip install -r requirements.txt
```

### 執行方式

#### 1. 命令列模式

```bash
python main.py --data data/raw/data.csv --periods 30 --freq D --plot
```

參數說明：
- `--data`：輸入數據路徑（必要）
- `--output`：輸出目錄（預設：`data/processed`）
- `--periods`：預測期數（預設：30）
- `--freq`：預測頻率（D=日，W=週，M=月）
- `--plot`：是否顯示圖表

#### 2. 互動式儀表板

```bash
streamlit run app/dashboard.py
```

## 🐳 Docker 開發環境

### 使用指令

```bash
# 構建映像
docker-compose build

# 啟動環境
docker-compose up -d

# 進入容器
docker-compose exec app bash

# 關閉環境
docker-compose down

# 清理映像
docker image prune -f
```

## 📊 應用範圍

- 中長期工廠用電預測（月/季/年）
- 可擴充整合：
  - 氣象資料
  - 假日因素
  - 工廠排程
- 目前支援：歷史用電 + 時間特徵（單變數預測）

## 📝 開發備註

- 直接比較模型在預測期間的績效
- 支援彈性預測週期設定
- 模組化架構設計，便於擴充

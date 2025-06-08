# 車流量偵測專案

本專案是一個基於 PyQt6、OpenCV 及 YOLO 的車流量偵測與計數工具，可擷取台灣高速公路監視器影像，進行車輛偵測、追蹤與流量計算。

## 功能簡介

- 擷取高速公路監視器即時影像
- 於畫面上繪製計數線
- 使用 YOLO 模型進行車輛偵測與追蹤
- 計算車輛跨越線段的數量與車流量
- 支援原始/標註影片回放

## 專案結構

```
Car_Detect.py         # 主程式
carUI.py              # PyQt UI 程式
carUI.ui              # PyQt Designer UI 檔
botsort.yaml          # 追蹤器設定檔
best.pt               # YOLO 權重檔（需自備）
raw_frames/           # 擷取的原始影格
annotated_frames/     # 標註後的影片
requirements.txt      # 依賴套件列表
README.md             # 本說明文件
```

## 執行方式

```
python Car_Detect.py
```

## 操作說明

1. 選擇監視器地點並擷取影像
2. 點擊「畫線」後於畫面上點選4個點（2條線），按 Enter 完成
3. 點擊「確認線條」
4. 點擊「開始」進行擷取、偵測與計數
5. 可於右側看到即時計數與車流量
6. 偵測完成後可回放原始或標註影片

"""阿斯拉量化系統 — ML 增強模組

提供多模型滾動預測引擎，輸出量化預測信號供 LLM 參考。

架構：
  base.py            — BasePredictor 抽象介面
  registry.py        — ModelRegistry（模型註冊中心）
  feature_engineer.py — 特徵工程 pipeline
  model_manager.py   — 模型版本管理、再訓練排程
  comparison.py      — 多模型比較引擎
  models/            — 各模型實作
"""

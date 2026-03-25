# My Buddy：基于神经同步动态演化（NSDC）架构的高自然度 TTS 引擎
**纯自研模型版 | 官方 README**


My Buddy 是一款专为数字人交互、具身智能及城市生活智能体设计的高自然度语音合成（TTS）系统。本项目创新性地提出了 NSDC (Neural-Sync Dynamic Conformer) 混合架构，旨在通过深度语义理解驱动声学演化，攻克传统 TTS 韵律机械、情感缺失的行业痛点。

---

## 核心技术亮点

**BERT-Brain 语义中枢**
基于 RoBERTa-WWM-Ext 预训练模型，通过深度 Transformer 结构解析文本全局语义，精准预测停顿、时长、重音及呼吸点，实现语义与韵律的深度绑定。

**NSDC 自研声学架构**
自研 Lite-Conformer 核心算子，融合 Self-Attention 全局建模能力与 1D 深度卷积局部特征提取能力，兼顾语气连贯性与发音清晰度，实现轻量化高性能推理。

**极致自然韵律**
采用 ResidualConv1DBlock 空洞卷积残差结构，大幅扩大时序感受野，精准捕捉人声细微纹理与呼吸起伏，彻底消除合成语音的“电音感”与机械感。

**广播级音质输出**
集成 BigVGAN 生成对抗网络声码器，支持 44.1kHz 高采样率音频生成，提供高保真、通透、自然的真人级听觉体验。

**零延迟交互看板**
配套前后端分离可视化管理界面，支持流式语音播报、实时合成与梅尔频谱动态监控，全流程离线可用。

---

## 系统架构

项目采用“感知-生成-还原”三层解耦架构，全模块自研、可独立迭代、支持本地离线部署。

**文本分析层 (The Brain)**
以 TextEncoder 与 ProsodyPredictor 为核心，通过 RoBERTa 提取 256 维语义特征，经由 1D CNN 韵律预测头输出停顿时长、重音、呼吸等标签，实现从字面理解到语境理解。

**声学建模层 (The Body)**
以自研 NSDC Conformer 为核心，通过时长调节器完成文本特征与音频帧对齐，经由 Lite-Conformer 与残差卷积模块处理，输出 80 维梅尔频谱。

**波形合成层 (The Voice)**
以 BigVGAN Vocoder 为核心，将梅尔频谱转换为高保真语音波形，RTF < 0.1，支持端侧实时推理。

---



## 快速上手

### 环境准备（需python3.10以上版本）
```bash
git clone https://github.com/your-username/MyBuddy.git
cd MyBuddy
pip install -r requirements.txt

## 模型训练
python models/bert_brain/train.py
python models/conformer/train.py

## 模型测试
python models/bert_brain/inference.py
python models/vocoder/inference.py
```
---
开源协议
本项目基于 Apache License 2.0 开源。NSDC 架构、Lite-Conformer、BERT-Brain 均为自主研发技术.

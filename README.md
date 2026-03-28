# My Buddy：基于神经同步动态级联Neural-Sync Dynamic Cascade（NSDC）架构的高自然度语音合成（TTS）引擎
**纯自研文本与声学建模版 | 官方 README**

My Buddy 是一套面向中文数字人交互与离线部署场景的语音合成系统。当前版本采用三段式架构：

- `BERT-Brain`：负责文本语义与韵律条件建模
- `Attention Acoustic`：负责通过 attention 对齐把文本条件映射成 log-mel 频谱
- `BigVGAN Vocoder`：负责把 mel 频谱还原为高保真波形



---

## 核心技术亮点

**BERT-Brain 语义中枢**  
基于 RoBERTa-WWM-Ext 提取 256 维文本语义特征，并预测停顿、停顿时长、token 级帧长、呼吸、事件和句级情感，为第二阶段提供结构化条件。

**Attention Acoustic v2**  
第二阶段采用 `token conditioning -> token encoder -> frame queries -> cross attention -> mel + postnet` 的 attention 声学建模路线，并加入 guided attention 约束，让对齐更稳定。不再强依赖 token 复制展开，而是通过 attention 学习文本到声学帧的对齐关系。

**显式韵律条件注入**  
第二阶段仍然显式接收第一阶段输出的 `pause / breath / token duration / event / emotion` 条件特征。也就是说，第一阶段 `bert_brain` 依然保留，而且它的韵律特征仍然会被第二阶段使用。

**广播级音质输出**  
集成 BigVGAN 声码器，支持 44.1kHz 音频生成，提供高保真语音输出能力。

---

## 系统架构

项目采用“理解-声学-还原”三层解耦架构，模块可独立训练、替换与调试。

**文本分析层 (The Brain)**  
`TextEncoder + ProsodyPredictor` 将文本转换为 256 维语义特征，并预测停顿、时长、呼吸、事件和情感条件。

**声学建模层 (The Body)**  
`Attention Acoustic` 先对 token 条件序列编码，再构造帧级 query，通过 cross-attention 对齐到文本条件，并用 guided attention loss 约束单调对齐，最后输出 128 维 log-mel 频谱。

**波形合成层 (The Voice)**  
`BigVGAN Vocoder` 将 mel 频谱还原为高保真语音波形。

---

**<span style="color:red">模型推理测试前请确认模型权重已下载并存放到指定文件夹</span>**

## 权重存放链接
modelscope: https://modelscope.cn/models/notnot/MyBuddy_NSDC_TTS

huggingFace: https://huggingface.co/Releer/MyBuddy_NSDC_TTS

## 代码存放链接
github：https://github.com/releerr/My_Buddy_TTS

---

## 快速上手
**<span style="color:red">需更改所有模型代码中标注“#TODO”部分的文件夹名称，替换成您的本地文件夹</span>**

### 环境准备（需 Python 3.10 以上）
```bash
git clone https://github.com/your-username/MyBuddy.git
cd MyBuddy
pip install -r requirements.txt
```

### 模型训练
```bash
python models/bert_brain/train.py
ACOUSTIC_STAGE=short python models/acoustic/train.py
ACOUSTIC_STAGE=medium python models/acoustic/train.py
ACOUSTIC_STAGE=full python models/acoustic/train.py
```

### 数据准备
```bash
python models/acoustic/scripts/prepare_acoustic_data.py
```

### 模型推理测试
```bash
python models/bert_brain/inference.py
python models/vocoder/inference.py
```

### 推荐训练节奏
第二阶段建议使用课程学习，而不是一开始直接全量：

- `short`：优先学习最容易的一批短样本，默认 `100` 条
- `medium`：扩大到默认 `500` 条，并自动从 `short` 阶段初始化
- `full`：最后再进入全量训练，并自动从 `medium` 阶段初始化

这三阶段使用的是同一个 `Attention Acoustic` 模型架构，只是训练数据规模和 checkpoint 不同。

---

## 开源协议
本项目基于 Apache License 2.0 开源。BERT-Brain、Attention Acoustic 与相关架构脚本均为本项目自研与迭代实现。

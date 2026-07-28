# EOPD-SR: Entity-Ontology and Path-Dependency Subgraph Retrieval for Knowledge Graph–Augmented Reasoning

**License:** [MIT](LICENSE)

## Data Access & Datasets

All four benchmark datasets used in this paper are publicly available. Below are the download links and acquisition instructions for each.

### Dataset Download Links

| Dataset | KG Backend | HuggingFace Link | Original Source |
|---------|-----------|------------------|-----------------|
| **WebQSP** | Freebase | [`rmanluo/RoG-webqsp`](https://huggingface.co/datasets/rmanluo/RoG-webqsp) | [GitHub: YihSun/WebQSP](https://github.com/YihSun/WebQSP) |
| **CWQ** | Freebase | [`rmanluo/RoG-cwq`](https://huggingface.co/datasets/rmanluo/RoG-cwq) | [GitHub: princeton-nlp/ComplexWebQuestions](https://github.com/princeton-nlp/ComplexWebQuestions) |
| **GrailQA** | Freebase | [`Salesforce/grailqa`](https://huggingface.co/datasets/Salesforce/grailqa) | [GitHub: salesforce/grailqa](https://github.com/salesforce/grailqa) |
| **EntityQuestions** | Wikidata | [`dwaraknath/entityquestions`](https://huggingface.co/datasets/dwaraknath/entityquestions) | [GitHub: princeton-nlp/EntityQuestions](https://github.com/princeton-nlp/EntityQuestions) |

### Automatic Download (Recommended)

The code uses HuggingFace `datasets` to automatically download and cache datasets on first use. No manual download is required:

```bash
# Datasets are auto-downloaded when you run the benchmark
python run_benchmark.py --table 1 --max_samples 100
```

### Manual Download (Optional)

Automatic download through the HuggingFace `datasets` library is the
recommended method (see above).

For offline or air-gapped environments, datasets must be converted to the
normalized JSONL format expected by `data/dataloader.py` and placed in
`./data/raw/` using the following naming convention:

```text
data/raw/webqsp_test.jsonl
data/raw/cwq_test.jsonl
data/raw/grailqa_val.jsonl
data/raw/entityquestions_test.jsonl
```

Each record must contain at least a `question` field and an `answer` or
`answers` field. Depending on the dataset, records may additionally contain
`answer_ids`, `topic_entities`, `topic_entity_names`, and `sparql_query`.

Directly downloading a HuggingFace dataset repository with
`huggingface-cli download` does not automatically produce these normalized
JSONL files. Therefore, automatic loading through `datasets` is recommended.

If you need to prepare files manually, you can use the HuggingFace `datasets`
library in a Python script:

```python
from datasets import load_dataset

ds = load_dataset("rmanluo/RoG-webqsp", split="test")
ds.to_json("data/raw/webqsp_test.jsonl")
```

### Freebase Knowledge Graph Setup (Required for WebQSP, CWQ, GrailQA)

Three of the four datasets (WebQSP, CWQ, GrailQA) require a local Freebase SPARQL endpoint. EntityQuestions uses the public Wikidata endpoint and does **not** require this setup.

#### Step 1: Install Virtuoso Open-Source Edition

```bash
# Option A: Docker (recommended — pin to a specific version tag)
docker pull openlink/virtuoso-opensource-7:7.2.11
mkdir -p ~/virtuoso-data

# Option B: Build from source (Ubuntu/Debian)
# See: https://github.com/openlink/virtuoso-opensource
sudo apt-get install virtuoso-opensource
```

> **Reproducibility note:** Record the exact Virtuoso version and Freebase dump
> checksum you use, because different snapshots may produce different SPARQL
> query results.

#### Step 2: Download Freebase Data Dump

The experiments reported in this paper use the Google Freebase RDF dump
(`freebase-rdf-latest.gz`, approximately 25 GB compressed). Download it from:

- **Primary:** https://developers.google.com/freebase (archived Google page)
- **Mirror:** https://github.com/google/freebase-rdf-dump

```bash
# Example (replace with the actual current URL from the sources above):
wget https://storage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz
```

> **Note:** Some datasets (WebQSP) also ship pre-processed Freebase subgraphs
> in their original repositories. Check
> [YihSun/WebQSP](https://github.com/YihSun/WebQSP/tree/main/data) for
> pre-extracted data if a full Freebase dump is not feasible.

#### Step 3: Load Freebase into Virtuoso

```bash
# Using Docker (use the same version tag as Step 1):
docker run -d \
  --name virtuoso \
  -p 1111:1111 \
  -p 8890:8890 \
  -v ~/virtuoso-data:/database \
  -v /path/to/freebase-rdf:/freebase \
  openlink/virtuoso-opensource-7:7.2.11

# Load the Freebase dump via isql (this may take several hours):
docker exec -it virtuoso isql 1111 dba dba
SQL> ld_dir('/freebase', 'freebase-rdf-latest.gz', 'http://freebase.org');
SQL> rdf_loader_run();
SQL> checkpoint;
```

#### Step 4: Verify Freebase Endpoint

```bash
# Test the SPARQL endpoint — should return JSON with 5 triples
curl -X POST http://localhost:8890/sparql \
  -H "Accept: application/json" \
  -d "query=SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"
```

#### Step 5: Configure Environment

```bash
# In your .env file:
FREEBASE_ENDPOINT=http://localhost:8890/sparql
```

> **Note:** If you do not have Freebase available, you can still run experiments on the **EntityQuestions** dataset only (uses public Wikidata endpoint, no setup required).

### Third-Party Licenses

The MIT License in this repository applies only to the EOPD-SR source code.
The benchmark datasets, Freebase, Wikidata, pretrained models, and external
services remain subject to their respective licenses and terms of use:

- **WebQSP** — [Apache License 2.0](https://github.com/YihSun/WebQSP)
- **CWQ** — [Apache License 2.0](https://github.com/princeton-nlp/ComplexWebQuestions)
- **GrailQA** — [Apache License 2.0](https://github.com/salesforce/grailqa)
- **EntityQuestions** — [MIT License](https://github.com/princeton-nlp/EntityQuestions)
- **Freebase** — [Creative Commons Attribution (CC-BY)](https://developers.google.com/freebase)
- **Wikidata** — [Creative Commons CC0](https://www.wikidata.org/wiki/Wikidata:Copyright)

## 项目概述

EOPD-SR 是一个无需训练的推理时框架，用于知识图谱增强的多跳问答。核心思想是将大型知识图谱压缩为小而精确的子图，以优化大语言模型（LLM）的推理。

### 核心架构

```
Question
    │
    ▼
┌─────────────────────────────────────────────────┐
│  E-Stage: Entity-Ontology Extraction            │
│  Step 1: LLM 实体提取                            │
│  Step 2: 候选实体检索 (BM25 + Fuzzy + Embedding) │
│  Step 3: 主题实体识别 (LLM 重排序)                │
│  Step 4: 本体路径子图提取 (BFS)                   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  P-Stage: Path-Dependency Retrieval             │
│  Step 1-3: 关系链接 (BM25 + LLM 重排序)          │
│  Step 4-5: 方向依赖可达集计算                     │
│  Step 6-7: 路径依赖剪枝                          │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  D-Stage: Dual-View Subgraph Assembly           │
│  Step 1: LLM 引导的单跳扩展                      │
│  Step 2: Dijkstra 重排序                         │
│  Step 3: 双视图合并                              │
│  Step 4: 预算感知剪枝                            │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Reasoning: GPT-4o 知识图谱推理                   │
│  子图 → 文本化 → LLM 推理 → 答案                  │
└─────────────────────────────────────────────────┘
                  │
                  ▼
              Answer
```

## 项目结构

```
EOPD-SR/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── test_setup.py                # 安装验证脚本
├── config.py                    # 配置管理（超参数、API 设置）
├── main.py                      # 主入口（单问题/批量评估）
├── run_benchmark.py             # 基准测试运行器（复现论文表格）
│
├── llm/
│   ├── __init__.py
│   └── llm_client.py            # LLM 客户端（GPT-4o-mini, Embeddings）
│
├── kg/
│   ├── __init__.py
│   ├── base.py                  # 知识图谱抽象接口
│   ├── freebase.py              # Freebase SPARQL 实现
│   └── wikidata.py              # Wikidata SPARQL 实现
│
├── data/
│   ├── __init__.py
│   └── dataloader.py            # 数据集加载器 (WebQSP, CWQ, GrailQA, EntityQuestions)
│
├── retrieval/
│   ├── __init__.py
│   ├── dense_retriever.py       # Bi-encoder 稠密检索器
│   ├── e_stage.py               # E-Stage: 实体-本体提取 (Algorithm 1)
│   ├── p_stage.py               # P-Stage: 路径依赖检索 (Algorithm 2)
│   └── d_stage.py               # D-Stage: 双视图子图组装
│
├── reasoning/
│   ├── __init__.py
│   ├── gpt_reasoner.py          # GPT-4o 推理模块
│   └── graph_rag_reasoner.py    # GraphRAG 推理模块
│
├── utils/
│   ├── __init__.py
│   └── graph_utils.py           # 图操作工具函数
│
└── evaluation/
    ├── __init__.py
    └── metrics.py               # 评估指标 (F1, Hits@1, ReCall@50)
```

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. 验证安装

```bash
python test_setup.py
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

**必需的环境变量：**
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API 密钥
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI 端点 URL

**可选（如果使用标准 OpenAI）：**
- `OPENAI_API_KEY`: OpenAI API 密钥

**Freebase 知识图谱（WebQSP/CWQ/GrailQA 数据集需要）：**
- 需要在本地安装 Virtuoso 并加载 Freebase 数据
- `FREEBASE_ENDPOINT`: SPARQL 端点（默认 `http://localhost:8890/sparql`）

### 4. 知识图谱后端配置

| 数据集 | 知识图谱 | 端点要求 |
|--------|---------|---------|
| WebQSP | Freebase | 本地 Virtuoso 服务器 |
| CWQ | Freebase | 本地 Virtuoso 服务器 |
| GrailQA | Freebase | 本地 Virtuoso 服务器 |
| EntityQuestions | Wikidata | 公共 SPARQL 端点（无需配置） |

## 使用方法

### 单问题问答

```bash
python main.py --question "Who is the president of France?" --kg wikidata
```

### 基准测试评估

```bash
# 在单个数据集上评估
python main.py --dataset webqsp --split test --max_samples 100

# 运行所有基准测试
python run_benchmark.py

# 复现论文 Table 1
python run_benchmark.py --table 1 --max_samples 500

# 复现论文 Table 2
python run_benchmark.py --table 2 --max_samples 500
```

### 使用不同模型

```bash
# 使用 GPT-4o（更强的推理能力）
python main.py --dataset webqsp --model gpt-4o

# 调整 token 预算
python main.py --dataset cwq --budget 6000
```

## 论文复现指南

### Table 1: WebQSP 和 CWQ 上的整体性能

```bash
python run_benchmark.py --table 1 --max_samples 1000
```

### Table 2: GrailQA 和 EntityQuestions 上的泛化性能

```bash
python run_benchmark.py --table 2 --max_samples 1000
```

### Table 5: GraphRAG 推理消融实验

```bash
# 使用 GraphRAG 作为推理后端
python main.py --dataset webqsp --model gpt-4o --budget 4000
```

### Table 6: Token 预算消融实验

```bash
for budget in 2000 4000 6000 8000; do
    python main.py --dataset webqsp --budget $budget --max_samples 200
done
```

### Table 7: 模型规模消融实验

```bash
for model in gpt-4o-mini gpt-4o; do
    python main.py --dataset webqsp --model $model --max_samples 200
done
```

## 超参数配置

所有超参数在 `config.py` 中定义，对应论文 Table 4：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `bm25_top_k` | 50 | BM25 候选实体数 |
| `embedding_top_k` | 50 | 稠密检索候选数 |
| `topic_entity_top_k` | 10 | 主题实体最大数量 |
| `max_ontology_depth` | 2 | 本体子图 BFS 深度 |
| `relation_llm_top_k` | 10 | LLM 重排序后保留的关系数 |
| `max_path_length` | 3 | 推理路径最大跳数 |
| `token_budget` | 4000 | 子图 token 预算 |
| `edge_budget` | 30 | 子图边数预算 |
| `path_length_penalty` | 0.1 | Dijkstra 路径长度惩罚系数 λ |

## 评估指标

| 指标 | 描述 |
|------|------|
| **F1** | 预测答案与金标答案的 token 级 F1 分数 |
| **Hits@1** | Top-1 预测是否匹配任意金标答案 |
| **ReCall@50** | Top-50 检索子图中金标答案的召回率 |
| **Avg. Graph Size** | 检索子图的平均三元组数量 |
| **Pct. Reduction** | 相比 Graph-only 基线的图大小缩减百分比 |

## 注意事项

1. **API 费用**：使用 OpenAI/Azure API 会产生费用。建议先用 `--max_samples 10` 测试。
2. **Freebase 部分**：WebQSP、CWQ、GrailQA 数据集需要本地 Freebase 端点。如果没有，可以使用 Wikidata 版本或跳过这些数据集。
3. **速率限制**：API 调用可能受到速率限制。代码内置了重试逻辑。
4. **缓存**：嵌入和 LLM 响应会缓存到 `./cache/` 目录，避免重复调用。

## 引用

```bibtex
@article{eposr2025,
  title={EOPD-SR: Entity-Ontology and Path-Dependency Subgraph Retrieval for Knowledge Graph–Augmented Reasoning},
  author={},
  journal={},
  year={2025}
}
```

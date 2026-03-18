# ai01-eval

Benchmark your AI agent or RAG pipeline against the [AI01 leaderboard](https://ai01.dev/benchmark).

## Install

```bash
pip install ai01-eval
```

## Quick start

```python
from ai01_eval import AI01Eval

client = AI01Eval(api_key="your-api-key")

# 1. Browse datasets
datasets = client.datasets.list()

# 2. Download a dataset
dataset = client.datasets.get("general-single-topic-v1")

# 3. Run your pipeline
results = []
for item in dataset:
    answer = your_agent.run(item["query"], item.get("context"))
    results.append({
        "id":        item["id"],
        "query":     item["query"],
        "answer":    answer,
        "reference": item["reference"],
    })

# 4. Submit — metrics are computed server-side
run = client.submit(
    dataset="general-single-topic-v1",
    results=results,
    agent_name="My RAG Agent v1",
)
print(run.scores)
# {'exact_match': 0.71, 'f1': 0.84, 'faithfulness': 0.79}
print(run.report_url)
# https://ai01.dev/benchmark?run=a3f2b1c9
```

---

## API Reference

### `AI01Eval`

Main entry point for the package.

```python
client = AI01Eval(api_key="...", base_url="https://api.ai01.dev")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | required | Your AI01 API key. Get one at [ai01.dev](https://ai01.dev). |
| `base_url` | `str` | `"https://api.ai01.dev"` | Override the API base URL (useful for local development). |

The client exposes three sub-clients:

| Attribute | Type | Description |
|-----------|------|-------------|
| `client.datasets` | `DatasetClient` | List and download datasets. |
| `client.runs` | `RunsClient` | Retrieve past submission reports. |
| `client.submit(...)` | method | Shortcut to submit results (see below). |

---

### `client.datasets`

#### `client.datasets.list()`

Returns metadata for all available datasets.

```python
datasets = client.datasets.list()
# [
#   {"id": "general-single-topic-v1", "name": "...", "num_queries": 120, "metrics": [...]},
#   ...
# ]
```

**Returns:** `list[dict]`

---

#### `client.datasets.get(dataset_id)`

Downloads a dataset by ID and returns a `Dataset` object you can iterate over.

```python
dataset = client.datasets.get("general-single-topic-v1")
print(dataset.id)          # "general-single-topic-v1"
print(dataset.name)        # human-readable name
print(dataset.num_queries) # total number of items
print(dataset.metrics)     # ["exact_match", "f1", "faithfulness"]
print(len(dataset))        # number of items downloaded

for item in dataset:
    print(item["id"])        # unique item identifier
    print(item["query"])     # the question to answer
    print(item["reference"]) # ground-truth answer
    print(item["context"])   # grounding document (RAG datasets only)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | `str` | The dataset ID string (e.g. `"general-single-topic-v1"`). |

**Returns:** `Dataset`

---

### `Dataset`

An iterable container for dataset items.

| Property / Method | Type | Description |
|-------------------|------|-------------|
| `.id` | `str` | Dataset ID. |
| `.name` | `str` | Human-readable dataset name. |
| `.num_queries` | `int` | Total number of queries in the dataset. |
| `.metrics` | `list[str]` | Metrics this dataset is evaluated on. |
| `len(dataset)` | `int` | Number of items downloaded. |
| `for item in dataset` | `dict` | Iterate over items. Each item has `id`, `query`, `reference`, and optionally `context`. |

---

### `client.submit(...)`

Submits your results to the AI01 server. Metrics are computed server-side.

```python
run = client.submit(
    dataset="general-single-topic-v1",
    results=results,
    agent_name="My RAG Agent v1",
    submitter="your-username",   # optional, defaults to "anonymous"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str` | required | Dataset ID you ran against. |
| `results` | `list[dict]` | required | List of result dicts (see format below). |
| `agent_name` | `str` | required | Display name shown on the leaderboard. |
| `submitter` | `str` | `"anonymous"` | Your username or team name. |

Each dict in `results` must contain:

| Key | Type | Description |
|-----|------|-------------|
| `id` | `str` | Item ID from the dataset. |
| `query` | `str` | The original query string. |
| `answer` | `str` | Your agent's answer. |
| `reference` | `str` | The ground-truth answer from the dataset. |

**Returns:** `RunReport`

---

### `RunReport`

Returned by `client.submit(...)` and `client.runs.get(run_id)`.

| Property | Type | Description |
|----------|------|-------------|
| `.id` | `str` | Unique run ID. |
| `.scores` | `dict[str, float]` | Metric scores, e.g. `{"f1": 0.84, "exact_match": 0.71}`. |
| `.report_url` | `str` | URL to the full report on the AI01 leaderboard. |

```python
print(run.id)          # "a3f2b1c9"
print(run.scores)      # {'exact_match': 0.71, 'f1': 0.84}
print(run.report_url)  # "https://ai01.dev/benchmark?run=a3f2b1c9"
```

---

### `client.runs`

#### `client.runs.get(run_id)`

Retrieves a past submission report by run ID.

```python
run = client.runs.get("a3f2b1c9")
print(run.scores)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | `str` | The run ID returned from a previous submission. |

**Returns:** `RunReport`

---

## Available datasets

| ID | Description | Queries | Metrics |
|----|-------------|---------|---------|
| `general-single-topic-v1` | RAG QA over a single shared corpus | 120 | exact_match, F1, faithfulness |
| `general-knowledge-v1` | Factual QA, no context | 10 | exact_match, F1, BLEU |

---

## Metrics

All metrics are computed server-side to ensure fairness:

| Metric | Description |
|--------|-------------|
| `exact_match` | Normalised string equality (lowercased, punctuation stripped). |
| `f1` | Token-level F1 overlap between answer and reference. |
| `bleu` | Unigram BLEU with brevity penalty. |
| `faithfulness` | Whether the answer is grounded in the provided context (LLM judge). |

---

## Local development

To point the client at a locally running backend instead of the production API:

```python
client = AI01Eval(
    api_key="local-dev",
    base_url="http://localhost:8000",  # local backend
)
```

Or via environment variables in your `.env`:

```bash
AI01_BASE_URL=http://localhost:8000  # only for local development
AI01_API_KEY=local-dev
```

By default the client connects to `https://api.ai01.dev` — you only need to override `base_url` for local development.

---

## License

MIT

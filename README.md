# Goal-Driven Multi-Agent Simulation Framework for Evaluating Recommender Systems under Data Scarcity

A simulation framework that uses goal-driven LLM agents inside a RecSim environment to generate synthetic user interaction logs for evaluating news recommender systems when real interaction data is unavailable.

## Overview

**Core idea**: Generate virtual news readers with explicit goals, run them as agents through a RecSim environment over a MIND news corpus, collect synthetic interaction logs, and validate them against real MIND user logs using two complementary evaluation approaches.

**Dataset**: [MIND](https://msnews.github.io/) (MIND-small: 50K users by default; MIND-large: 1M via config flag)  
**Simulation**: [RecSim](https://github.com/google-research/recsim) (actual library, Gym-compatible environment)  
**LLM Judge**: `google/gemma-4-E4B-it` via HuggingFace Inference API  
**Agent language**: Bilingual — English + French (configurable per persona archetype)

## Project Structure

```
data/               MIND loader + FAISS index builder
recsim_env/         RecSim AbstractDocument / UserState / UserModel / Response
simulation/         Bilingual goal-driven agent + user generator + SQLite persistence + runner
recommender/        FAISS profile recommender
evaluation/
  fidelity.py       Approach A: CTR-KL, Wasserstein session-length, ILD, drift
  approach_a.py     Approach A: Replay / distribution matching
  approach_b.py     Approach B: NCF train-on-synthetic / test-on-real NDCG@10
export/             TREC qrels, MSNEWS behaviors.tsv, RecSim SequenceExample
api/                Optional FastAPI IR service
configs/            mind_small.yaml / mind_large.yaml
notebooks/          01_data_prep · 02_simulation · 03_evaluation
llm/                Gemma judge + bilingual Jinja2 prompt templates
```

## Quick Start

```bash
pip install -r requirements.txt

# Set HuggingFace token
echo "HF_TOKEN=hf_..." > .env

# 1. Prepare data (download MIND from https://msnews.github.io/ first)
jupyter notebook notebooks/01_data_prep.ipynb

# 2. Run simulation
jupyter notebook notebooks/02_simulation.ipynb

# 3. Evaluate
jupyter notebook notebooks/03_evaluation.ipynb

# MIND-large: swap the config file in each notebook
# config = yaml.safe_load(open('configs/mind_large.yaml'))
```

## Evaluation Approaches

| Approach | Question | Method | Key metric |
|---|---|---|---|
| **A** | Does the synthetic log resemble real MIND? | Distribution matching | CTR KL-divergence, Wasserstein session length |
| **B** | Does synthetic training data generalise? | NCF on synthetic → test on real | NDCG@10, Recall@10 |

## Persona Archetypes

| Archetype | Language | Share |
|---|---|---|
| `breaking_news_follower` | EN | 28% |
| `topic_specialist` | EN/bilingual | 22% |
| `casual_browser` | EN | 20% |
| `sentiment_tracker` | FR | 15% |
| `deep_reader` | FR | 15% |

Shares are recalibrated from real MIND K-means user clusters — see `simulation/user_generator.py`.
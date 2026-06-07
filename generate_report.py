"""Generate the project technical report as a PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUT = "report.pdf"
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Goal-Driven Multi-Agent Simulation Framework",
    author="Hatem Amine",
)

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

TITLE  = S("Title2",  "Title",  fontSize=20, textColor=colors.HexColor("#1a237e"),
           spaceAfter=6, alignment=TA_CENTER)
SUB    = S("Sub",     "Normal", fontSize=12, textColor=colors.HexColor("#455a64"),
           spaceAfter=14, alignment=TA_CENTER)
DATE   = S("Date",    "Normal", fontSize=10, textColor=colors.grey,
           spaceAfter=4, alignment=TA_CENTER)
H1     = S("H1",      "Heading1", fontSize=14, textColor=colors.HexColor("#1a237e"),
           spaceBefore=18, spaceAfter=6, borderPad=2)
H2     = S("H2",      "Heading2", fontSize=12, textColor=colors.HexColor("#283593"),
           spaceBefore=12, spaceAfter=4)
H3     = S("H3",      "Heading3", fontSize=11, textColor=colors.HexColor("#37474f"),
           spaceBefore=8, spaceAfter=3)
BODY   = S("Body",    "Normal",   fontSize=10, leading=15, alignment=TA_JUSTIFY,
           spaceAfter=6)
CODE   = S("Code",    "Code",     fontSize=8.5, leading=13, leftIndent=14,
           backColor=colors.HexColor("#f5f5f5"), spaceAfter=6)
CAPTION= S("Cap",     "Normal",   fontSize=9, textColor=colors.grey,
           alignment=TA_CENTER, spaceAfter=8)
BULLET = S("Bullet",  "Normal",   fontSize=10, leading=14, leftIndent=14,
           spaceAfter=3)

def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#90caf9"), spaceAfter=6)
def sp(n=6): return Spacer(1, n)
def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def h3(t): return Paragraph(t, H3)
def p(t):  return Paragraph(t, BODY)
def code(t): return Paragraph(t.replace("\n","<br/>").replace(" ","&nbsp;"), CODE)
def bull(items):
    return ListFlowable(
        [ListItem(Paragraph(i, BULLET), leftIndent=18, bulletColor=colors.HexColor("#1565c0")) for i in items],
        bulletType="bullet", leftIndent=10, spaceAfter=6,
    )

def section_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0), 9),
        ("BACKGROUND",  (0,1),(-1,-1), colors.HexColor("#e8eaf6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#e8eaf6"), colors.white]),
        ("FONTSIZE",    (0,1),(-1,-1), 9),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#9fa8da")),
        ("ALIGN",       (0,0),(-1,-1), "LEFT"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
    ]))
    return t

# ── Content ───────────────────────────────────────────────────────────────────
story = []

# Cover
story += [
    sp(40),
    Paragraph("Goal-Driven Multi-Agent Simulation Framework", TITLE),
    Paragraph("for Evaluating Recommender Systems under Data Scarcity", TITLE),
    sp(10),
    HRFlowable(width="60%", thickness=2, color=colors.HexColor("#1565c0"), hAlign="CENTER"),
    sp(12),
    Paragraph("Technical Report", SUB),
    Paragraph("Hatem Amine · ESTIN · 2024", DATE),
    sp(6),
    Paragraph("hatem@estin.dz", DATE),
    PageBreak(),
]

# ── 1. Abstract ───────────────────────────────────────────────────────────────
story += [
    h1("1. Abstract"),
    hr(),
    p("Recommender systems depend heavily on large-scale user interaction logs for training "
      "and evaluation. In many real-world scenarios such data is unavailable due to cold-start "
      "conditions, privacy regulations, or domain novelty. This work presents a <b>Goal-Driven "
      "Multi-Agent Simulation Framework</b> that addresses this data-scarcity problem by "
      "generating synthetic interaction logs through LLM-powered virtual news readers operating "
      "inside a RecSim environment over the Microsoft News Dataset (MIND)."),
    p("Virtual agents are assigned explicit reading goals and persona archetypes. They interact "
      "with a FAISS-backed news retrieval system and make decisions guided by a locally-running "
      "Gemma-4 language model judge. The synthetic logs are validated against real MIND "
      "interaction data through two complementary evaluation approaches: <b>Approach A</b> "
      "(distribution matching / fidelity metrics) and <b>Approach B</b> (train-on-synthetic / "
      "test-on-real NDCG@10 via Neural Collaborative Filtering)."),
    sp(8),
]

# ── 2. Problem Statement ──────────────────────────────────────────────────────
story += [
    h1("2. Problem Statement"),
    hr(),
    p("Standard news recommendation pipelines require click logs to: (a) train ranking models, "
      "(b) compute offline evaluation metrics such as NDCG and CTR, and (c) study user behaviour "
      "patterns. When no historical log exists — for example at system launch, after a major "
      "topic shift, or in low-resource language settings — these pipelines fail entirely."),
    p("The core research question is: <i>Can we generate synthetic interaction logs that are "
      "both faithful to real user behaviour and useful as training data for recommender "
      "systems?</i>"),
    h2("2.1 Design Constraints"),
    bull([
        "<b>No real log available at generation time.</b> The simulator must produce plausible "
        "behaviour from scratch using only the news corpus.",
        "<b>Diversity of user types.</b> Real MIND users exhibit at least five behavioural "
        "clusters; the simulation must reproduce this distribution.",
        "<b>Bilingual corpus.</b> MIND contains English articles; the agent framework supports "
        "both English and French personas.",
        "<b>Reproducibility.</b> All LLM calls are cached; results are fully deterministic "
        "given the same random seed.",
    ]),
    sp(8),
]

# ── 3. Architecture Overview ──────────────────────────────────────────────────
story += [
    h1("3. Architecture Overview"),
    hr(),
    p("The framework is composed of six layers that interact in a feed-forward pipeline:"),
    sp(4),
    section_table(
        [
            ["Layer", "Module", "Responsibility"],
            ["Data", "data/mind_loader.py\ndata/news_preprocessor.py",
             "Load MIND TSV files, build FAISS index, cluster real users"],
            ["RecSim Environment", "recsim_env/",
             "Gym-compatible simulation loop: document sampling, user state, responses"],
            ["LLM Judge", "llm/judge.py\nllm/prompts/",
             "Local Gemma-4 inference: relevance scoring, goal-progress assessment, persona generation"],
            ["Simulation", "simulation/",
             "Agent policy, virtual-user generator, SQLite persistence, parallel runner"],
            ["Evaluation", "evaluation/",
             "Approach A fidelity metrics, Approach B NCF train/test"],
            ["Export", "export/",
             "TREC qrels, MIND behaviors.tsv, RecSim TFRecords"],
        ],
        col_widths=[3.2*cm, 4.8*cm, 8.5*cm],
    ),
    sp(10),
]

# ── 4. RecSim Role ────────────────────────────────────────────────────────────
story += [
    h1("4. The Role of RecSim"),
    hr(),
    p("RecSim (Ie et al., 2019) is Google's open-source library for building simulated "
      "recommendation environments. It defines a clean interface between the <i>recommender</i> "
      "(agent) and the <i>user</i> (environment) through a set of abstract base classes modelled "
      "on the OpenAI Gym API."),

    h2("4.1 Why RecSim?"),
    bull([
        "<b>Formal RL interface.</b> RecSim exposes reset() / step(slate) / reward semantics, "
        "enabling the agent to interact with the environment in a standard reinforcement-learning "
        "loop — which is exactly the session-based read/click cycle.",
        "<b>Separation of concerns.</b> Document sampling, user state evolution, and response "
        "generation are cleanly decoupled, making each component independently testable.",
        "<b>Multi-step sessions.</b> A single call to env.step(slate) advances one step; "
        "looping until is_terminal() naturally produces a complete reading session.",
        "<b>Reproducibility.</b> All random state flows through a seeded numpy RNG, so "
        "simulation runs are deterministic.",
    ]),

    h2("4.2 In-Repo Shim (recsim_compat.py)"),
    p("The official recsim package fails on Python ≥ 3.10 because it imports "
      "<i>tf.estimator.SessionRunHook</i>, which was removed in TensorFlow 2. Rather than "
      "downgrading Python or TensorFlow, we provide a 200-line pure-Python shim "
      "(<b>recsim_env/recsim_compat.py</b>) that re-implements all required abstract base "
      "classes with no TF dependency:"),
    bull([
        "AbstractDocument / AbstractDocumentSampler",
        "AbstractUserState / AbstractResponse",
        "AbstractUserSampler / AbstractUserModel",
        "Environment — reset(), step(), _sample_candidates()",
        "RecSimGymEnv — wraps Environment with a custom reward_aggregator",
    ]),
    p("All five recsim_env/ modules import exclusively from this shim. The rest of the "
      "codebase is unaffected."),

    h2("4.3 RecSim Step Loop"),
    p("Each simulation step proceeds as follows:"),
    section_table(
        [
            ["Step", "RecSim Component", "What Happens"],
            ["1. reset()", "Environment", "Calls user_model.reset() → samples new MindUserState; "
             "calls _sample_candidates() → doc_sampler prefetches FAISS results for current query"],
            ["2. Agent selects slate", "GoalDrivenAgent", "LLM decision prompt → returns list of "
             "candidate indices; cosine-similarity fallback if LLM fails"],
            ["3. step(slate)", "MindUserModel", "simulate_response(): LLM judge scores each document "
             "for relevance; Bernoulli samples clicks; updates interest vector"],
            ["4. update_state()", "MindUserModel", "Ticks fatigue +0.05; increments step counter"],
            ["5. is_terminal()", "MindUserState", "Returns True when fatigue ≥ 0.9 or "
             "steps ≥ session_budget"],
            ["6. Reward", "reward_aggregator", "Sum of relevance scores for clicked documents "
             "(dense reward signal)"],
        ],
        col_widths=[2.5*cm, 3.8*cm, 10.2*cm],
    ),
    sp(10),
]

# ── 5. Component Interactions ─────────────────────────────────────────────────
story += [
    h1("5. Component Interactions"),
    hr(),

    h2("5.1 Data Layer → RecSim"),
    p("Before any simulation begins, <b>news_preprocessor.py</b> encodes all MIND articles "
      "with <i>all-MiniLM-L6-v2</i> (384-d sentence embeddings) and stores a FAISS "
      "IndexFlatIP for inner-product search. At each session reset, "
      "<b>MindDocumentSampler.set_query()</b> fires a FAISS search against the current "
      "user goal, pre-loading up to 200 topically relevant articles as the candidate pool. "
      "Within a session, each env.step() draws from this pool in order (and replenishes it "
      "from FAISS when the agent updates its query)."),

    h2("5.2 LLM Judge → RecSim Responses"),
    p("Inside <b>MindUserModel.simulate_response()</b> — the central RecSim callback — the "
      "Gemma-4 judge is invoked once per document per step. It receives the user's goal, "
      "role, and the article title/abstract, and returns a relevance score in [0, 1]. "
      "This score drives a logistic click probability:"),
    p("<i>P(click) = sigmoid(6 · (relevance − 0.4)) − 0.25 · fatigue</i>"),
    p("All judge calls are cached in SQLite by <i>sha256(goal[:120]) × doc_id × lang × mode</i>, "
      "so repeated (goal, article) pairs across users cost zero inference time."),

    h2("5.3 Agent Policy → Environment"),
    p("<b>GoalDrivenAgent</b> implements the RL policy. At each step it:"),
    bull([
        "Reads the observation dict (user interest vector + doc candidate list) from RecSim.",
        "Calls the LLM decision prompt (<i>decision_en.j2</i> or <i>decision_fr.j2</i>) to "
        "select up to slate_size indices and obtain an updated search query.",
        "Falls back to cosine-similarity ranking between the interest vector and candidate "
        "embeddings if LLM generation fails.",
        "Returns the slate as a numpy index array to env.step().",
    ]),
    p("After each step the agent calls get_next_query() and passes the result to "
      "doc_sampler.set_query(), steering future candidate retrieval toward the agent's "
      "evolving information need."),

    h2("5.4 Persistence Layer"),
    p("The simulation runner stores every interaction in an SQLite database "
      "(<b>simulation/db.py</b>) with the following schema:"),
    section_table(
        [
            ["Table", "Key Columns", "Purpose"],
            ["virtual_users", "user_id, archetype, language_pref, goal, role",
             "One row per generated virtual user"],
            ["sessions", "session_id, user_id, session_num, total_clicks, final_fatigue, goal_progress",
             "One row per user × session"],
            ["interactions", "session_id, step, position, news_id, clicked, dwell_time, relevance, fatigue",
             "One row per document shown; the synthetic interaction log"],
        ],
        col_widths=[3.2*cm, 6.0*cm, 7.3*cm],
    ),
    sp(10),
]

# ── 6. User Model ─────────────────────────────────────────────────────────────
story += [
    h1("6. Virtual User Model"),
    hr(),

    h2("6.1 Persona Archetypes"),
    p("Five archetypes calibrated from K-means clustering of real MIND users:"),
    section_table(
        [
            ["Archetype", "Language", "Share", "Reading Style", "Session Budget"],
            ["breaking_news_follower", "EN", "28 %", "skimmer",     "25 articles"],
            ["topic_specialist",       "EN", "22 %", "deep_reader", "15 articles"],
            ["casual_browser",         "EN", "20 %", "balanced",    "20 articles"],
            ["sentiment_tracker",      "FR", "15 %", "skimmer",     "20 articles"],
            ["deep_reader",            "FR", "15 %", "deep_reader", "12 articles"],
        ],
        col_widths=[5.0*cm, 2.2*cm, 1.8*cm, 3.0*cm, 3.5*cm],
    ),
    sp(6),

    h2("6.2 User State (MindUserState)"),
    bull([
        "<b>interest_vector</b> (384-d): weighted running mean of clicked document embeddings; "
        "updated as α·doc_emb + (1−α)·interest with α = 0.3 × relevance.",
        "<b>fatigue</b> (0–1): incremented by 0.05 per step; session terminates at 0.9.",
        "<b>goal</b>: natural-language reading objective generated by Gemma-4 via the persona "
        "prompt template.",
        "<b>current_query</b>: evolves during the session as the agent refines its "
        "information need.",
    ]),
    sp(8),
]

# ── 7. Evaluation ─────────────────────────────────────────────────────────────
story += [
    h1("7. Evaluation Framework"),
    hr(),

    h2("7.1 Approach A — Distribution Matching"),
    p("Approach A measures how closely the synthetic log resembles real MIND user behaviour "
      "without requiring model training. Six metrics are computed:"),
    section_table(
        [
            ["Metric", "Definition", "Ideal"],
            ["CTR KL-divergence", "KL(P_sim_categories ‖ P_real_categories)", "→ 0"],
            ["Session-length Wasserstein", "W₁ distance between click-per-session histograms", "→ 0"],
            ["Category entropy gap", "|H(real categories) − H(sim categories)|", "→ 0"],
            ["Position-bias ρ", "Spearman correlation between position and CTR", "Negative (like real users)"],
            ["ILD", "Mean pairwise (1 − cosine) of clicked embeddings", "Higher → more diverse"],
            ["Replay NDCG@k", "NDCG of real impressions ranked by position order", "Baseline reference"],
        ],
        col_widths=[4.5*cm, 7.0*cm, 4.5*cm],
    ),
    sp(6),

    h2("7.2 Approach B — Train on Synthetic / Test on Real"),
    p("Approach B trains a Neural Collaborative Filtering (NCF) model on the synthetic log "
      "and evaluates it on real MIND test impressions. This directly answers: <i>is the "
      "synthetic data useful as a training set?</i>"),
    p("Because synthetic users (vuser_0001 …) and real MIND users (U123456 …) have disjoint "
      "IDs, a direct user-lookup evaluation is impossible. Instead we apply "
      "<b>zero-shot item-embedding transfer</b>:"),
    bull([
        "NCF learns item embeddings that encode co-click structure from synthetic interactions.",
        "For each real test user, a query vector is formed as the mean of their click-history "
        "item embeddings (items seen during synthetic training).",
        "Impression candidates are ranked by cosine similarity to the query.",
        "NDCG@10 and Recall@10 are computed on the ranked list.",
    ]),
    p("An ablation study varies N ∈ {100, 500, 1000} synthetic users to measure how "
      "metric quality scales with synthetic data volume."),

    h2("7.3 NCF Architecture"),
    p("GMF + MLP fusion (He et al., 2017) with BPR pairwise ranking loss:"),
    section_table(
        [
            ["Component", "Detail"],
            ["GMF branch",   "Element-wise product of user and item embeddings (emb_dim=32)"],
            ["MLP branch",   "Concatenated user+item embeddings → FC layers (64→32→16) with ReLU"],
            ["Output layer", "Linear(emb_dim + 16, 1) — predicts preference score"],
            ["Loss",         "BPR: −log σ(score_pos − score_neg) over randomly sampled negatives"],
            ["Optimiser",    "Adam, lr=1e-3, batch_size=256, 10 epochs"],
        ],
        col_widths=[3.5*cm, 13.0*cm],
    ),
    sp(10),
]

# ── 8. Data Flow Diagram (text) ───────────────────────────────────────────────
story += [
    h1("8. End-to-End Data Flow"),
    hr(),
    p("The full pipeline from raw MIND files to paper-ready metrics:"),
    sp(4),
    section_table(
        [
            ["Stage", "Input", "Output", "Module"],
            ["1. Index",      "news.tsv",
             "FAISS index + SQLite news DB", "news_preprocessor.py"],
            ["2. Cluster",    "behaviors.tsv",
             "Archetype distribution {name: fraction}", "mind_loader.py"],
            ["3. Generate users", "Archetype distribution",
             "List of virtual user profiles (goal, role, topics, …)", "user_generator.py"],
            ["4. Simulate",   "User profiles + FAISS index",
             "SQLite interactions DB (clicked, relevance, dwell_time, …)", "runner.py + RecSim"],
            ["5. Approach A", "Interactions DB + real behaviors",
             "Fidelity report dict", "approach_a.py"],
            ["6. Approach B", "Interactions DB + real behaviors",
             "NCF model; NDCG@10, Recall@10 per ablation size", "approach_b.py"],
            ["7. Export",     "Interactions DB",
             "TREC qrels.txt, behaviors.tsv, TFRecords (optional)", "export/"],
        ],
        col_widths=[2.8*cm, 3.5*cm, 5.2*cm, 5.0*cm],
    ),
    sp(10),
]

# ── 9. Technology Stack ───────────────────────────────────────────────────────
story += [
    h1("9. Technology Stack"),
    hr(),
    section_table(
        [
            ["Concern", "Library / Tool", "Version"],
            ["Simulation loop",   "recsim_compat (in-repo shim)", "custom"],
            ["Gym interface",     "gymnasium",                     "≥ 0.29"],
            ["News embeddings",   "sentence-transformers all-MiniLM-L6-v2", "≥ 2.2"],
            ["Semantic search",   "FAISS IndexFlatIP",             "faiss-cpu ≥ 1.7.4"],
            ["LLM judge",         "google/gemma-4-E4B-it (local)", "transformers ≥ 4.45"],
            ["4-bit quantisation","bitsandbytes (optional)",       "≥ 0.43"],
            ["Prompt templating", "Jinja2",                        "≥ 3.1"],
            ["RS training",       "PyTorch (NCF)",                 "≥ 2.0"],
            ["Data wrangling",    "pandas + scikit-learn",         "≥ 2.0 / 1.3"],
            ["Statistics",        "scipy",                         "≥ 1.11"],
            ["Progress display",  "tqdm",                          "≥ 4.66"],
            ["Persistence",       "SQLite (stdlib)",               "—"],
            ["Config",            "PyYAML + python-dotenv",        "≥ 6.0 / 1.0"],
        ],
        col_widths=[4.0*cm, 7.5*cm, 5.0*cm],
    ),
    sp(10),
]

# ── 10. Limitations & Future Work ─────────────────────────────────────────────
story += [
    h1("10. Limitations and Future Work"),
    hr(),
    bull([
        "<b>LLM inference cost.</b> Each (goal, article) pair requires one Gemma forward pass. "
        "The SQLite cache mitigates this for repeated pairs, but cold-start runs over large "
        "corpora remain slow without a GPU.",
        "<b>English-centric embeddings.</b> all-MiniLM-L6-v2 is primarily trained on English "
        "text; French article retrieval quality may be lower. A multilingual model "
        "(e.g. paraphrase-multilingual-MiniLM-L12-v2) would improve FR persona accuracy.",
        "<b>Click model simplicity.</b> The logistic click function does not model "
        "position bias, novelty, or contextual effects beyond fatigue. A more realistic "
        "cascade click model could improve Approach A fidelity.",
        "<b>Approach B item coverage.</b> Zero-shot transfer only benefits real users whose "
        "click history overlaps with articles the NCF has seen. Coverage grows with more "
        "synthetic users and more simulation sessions.",
        "<b>Single-domain.</b> The current framework is tested only on MIND (news). "
        "Extension to e-commerce or academic paper recommendation is straightforward "
        "given the modular design.",
    ]),
    sp(8),
]

# ── 11. References ────────────────────────────────────────────────────────────
story += [
    h1("11. References"),
    hr(),
    bull([
        "Ie, E. et al. (2019). <i>RecSim: A Configurable Simulation Platform for "
        "Recommender Systems.</i> arXiv:1909.04847.",
        "Wu, F. et al. (2020). <i>MIND: A Large-scale Dataset for News Recommendation.</i> "
        "ACL 2020.",
        "He, X. et al. (2017). <i>Neural Collaborative Filtering.</i> WWW 2017.",
        "Reimers, N. & Gurevych, I. (2019). <i>Sentence-BERT: Sentence Embeddings using "
        "Siamese BERT-Networks.</i> EMNLP 2019.",
        "Johnson, J. et al. (2019). <i>Billion-scale similarity search with GPUs "
        "(FAISS).</i> IEEE TBBM.",
        "Team, G. (2024). <i>Gemma 2: Improving Open Language Models at a Practical Size.</i> "
        "Google DeepMind.",
    ]),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"Report written to {OUT}")

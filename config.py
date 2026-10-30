"""config.py - every tunable knob lives here so experiments are reproducible."""
from pathlib import Path                      # Path gives OS-independent file paths

ROOT = Path(__file__).parent                  # folder that contains this file
DATA_DIR = ROOT / "data"                      # where scraped docs + LLM cache live
RESULTS_DIR = ROOT / "results"                # where CSVs and figures are written
DATA_DIR.mkdir(exist_ok=True)                 # create the folders if they don't exist yet
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_PATH = DATA_DIR / "docs.json"            # output of scrape.py, input of run_experiment.py
CACHE_PATH = DATA_DIR / "llm_cache.sqlite"    # disk cache so crashed runs can resume
RESULTS_CSV = RESULTS_DIR / "gossip_results.csv"  # one row per (doc, rule, seed, round)

# ---------------------------------------------------------------- scraping
SEED_URLS = [                                 # GOV.UK pages that are number/date heavy (easy to verify)
    "https://www.gov.uk/income-tax-rates",
    "https://www.gov.uk/national-minimum-wage-rates",
    "https://www.gov.uk/vat-rates",
    "https://www.gov.uk/child-benefit-rates",
    "https://www.gov.uk/new-state-pension/what-youll-get",
    "https://www.gov.uk/universal-credit/eligibility",
    "https://www.gov.uk/maternity-pay-leave/pay",
    "https://www.gov.uk/statutory-sick-pay",
    "https://www.gov.uk/council-tax-bands",
    "https://www.gov.uk/stamp-duty-land-tax/residential-property-rates",
    "https://www.gov.uk/capital-gains-tax/allowances",
    "https://www.gov.uk/marriage-allowance",
    "https://www.gov.uk/benefit-cap",
    "https://www.gov.uk/inheritance-tax",
]
USER_AGENT = "academic-gossip-llm-study/0.1 (contact: you@university.ac.uk)"  # be identifiable + polite
REQUEST_DELAY_S = 1.5                         # pause between requests so we don't hammer the site
MAX_DOC_WORDS = 350                           # truncate docs so one note fits a small model's context
MIN_SENTENCES = 12                            # need enough sentences to split across agents
MIN_ANCHORS = 6                               # need enough numbers/dates/acronyms to measure loss

# ------------------------------------------------------------------ ollama
OLLAMA_HOST = "http://localhost:11434"        # default Ollama server address
CHAT_MODEL = "llama3.2:3b"                    # any model you've `ollama pull`ed
EMBED_MODEL = "nomic-embed-text"              # embedding model for semantic metrics
TEMPERATURE = 0.3                             # low-but-nonzero: realistic yet fairly stable
MAX_NEW_TOKENS = 700                          # cap on generated tokens per call

# ------------------------------------------------------------------- gossip
N_AGENTS = 5                                  # number of nodes in the network
N_ROUNDS = 6                                  # gossip rounds (each agent initiates one exchange per round)
SEEDS = [0, 1]                                # independent repetitions (peer choice + LLM sampling)
MAX_NOTE_WORDS = 400                          # word budget stated in merge prompts
RULES = ["lww", "llm_naive", "llm_anchor", "llm_repair", "claim_union"]  # reconciliation rules to compare
COVERAGE_THRESHOLD = 0.80                     # cosine >= this => a source sentence counts as "covered"

# ------------------------------------------------------ gossip protocol (fixed)
# One protocol is used throughout (see gossip.py): push-pull exchange on a fully connected network,
# each agent contacting one random peer per round. Only the merge rule (RULES above) varies.

# ------------------------------------------------------- hypothesis testing
EPSILON = 0.10                 # tolerable loss: up to 10% of the source's verifiable details may be lost
ALPHA = 0.05                   # significance level (Holm-adjusted where several tests are run together)
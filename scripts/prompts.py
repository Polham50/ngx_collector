"""
NGX-FND Prompt Templates
=========================
Structured prompts for zero-shot, few-shot sentiment extraction
and forward guidance detection on Nigerian financial narratives.

All prompts are designed to return structured JSON for reproducible evaluation.
"""

# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analyst specializing in African capital markets, \
with deep expertise in Nigerian corporate disclosures and the Nigerian Exchange (NGX/NSE). \
You analyze excerpts from annual reports, earnings releases, and management commentary \
published by Nigerian listed companies.

Your task is to analyze financial narrative passages and return structured assessments. \
Always respond with valid JSON only — no preamble, no markdown, no explanation outside the JSON."""


# ── Task A: Sentiment Classification ──────────────────────────────────────────

SENTIMENT_ZERO_SHOT = """Analyze the sentiment of the following passage from a Nigerian corporate financial disclosure.

PASSAGE:
{passage}

Return ONLY a JSON object with this exact structure:
{{
  "sentiment": "<positive|negative|neutral>",
  "intensity": "<mild|moderate|strong>",
  "rationale": "<one sentence explaining your label>",
  "key_phrase": "<the single most sentiment-bearing phrase in the passage>"
}}

Definitions:
- positive: growth, improvement, optimism, strong performance, increased revenues/profits
- negative: decline, loss, challenges, headwinds, reduced performance
- neutral: factual/descriptive, balanced, no clear directional tone
- mild: slight lean, hedged language (e.g. "marginally improved")
- moderate: clear but measured (e.g. "recorded significant growth")
- strong: emphatic, unambiguous (e.g. "exceptional performance", "severe deterioration")"""


SENTIMENT_THREE_SHOT = """Analyze the sentiment of the following passage from a Nigerian corporate financial disclosure.

Here are three labeled examples to guide your assessment:

EXAMPLE 1 (positive, moderate):
"The Group recorded a 23% growth in gross earnings to N847 billion, driven by strong performance across our retail and corporate banking segments. Net interest income improved significantly as we benefited from the high interest rate environment."
→ {{"sentiment": "positive", "intensity": "moderate", "rationale": "Clear earnings growth with specific percentage cited across multiple segments", "key_phrase": "23% growth in gross earnings"}}

EXAMPLE 2 (negative, strong):
"The year under review was exceptionally challenging. Persistent foreign exchange volatility, escalating energy costs, and weak consumer purchasing power resulted in a significant contraction in volumes and margins across all business units."
→ {{"sentiment": "negative", "intensity": "strong", "rationale": "Multiple compounding headwinds with strong language about contraction across all units", "key_phrase": "significant contraction in volumes and margins across all business units"}}

EXAMPLE 3 (neutral, mild):
"During the financial year ended 31 December 2023, the Board of Directors met four times to review the strategic direction of the Company. Attendance at all meetings was satisfactory and in line with the provisions of the Company's Articles of Association."
→ {{"sentiment": "neutral", "intensity": "mild", "rationale": "Purely procedural/governance content with no financial performance signal", "key_phrase": "Board of Directors met four times"}}

Now analyze this passage:

PASSAGE:
{passage}

Return ONLY a JSON object with this exact structure:
{{
  "sentiment": "<positive|negative|neutral>",
  "intensity": "<mild|moderate|strong>",
  "rationale": "<one sentence explaining your label>",
  "key_phrase": "<the single most sentiment-bearing phrase in the passage>"
}}"""


SENTIMENT_FIVE_SHOT = """Analyze the sentiment of the following passage from a Nigerian corporate financial disclosure.

Here are five labeled examples:

EXAMPLE 1 (positive, strong):
"We are delighted to report a transformational year for the Group. Profit before tax surged 187% to N1.2 trillion, setting an all-time record. This outstanding performance reflects the disciplined execution of our strategic priorities and the resilience of our diversified business model."
→ {{"sentiment": "positive", "intensity": "strong", "rationale": "Record-breaking profit with superlative language and high percentage growth", "key_phrase": "Profit before tax surged 187% to N1.2 trillion"}}

EXAMPLE 2 (positive, moderate):
"Revenue for the year increased by 18% to N65.4 billion supported by volume growth in our core product categories and improved pricing. Operating profit margin expanded by 210 basis points reflecting efficiency gains from our cost optimisation programme."
→ {{"sentiment": "positive", "intensity": "moderate", "rationale": "Solid revenue and margin growth with specific metrics but measured tone", "key_phrase": "Operating profit margin expanded by 210 basis points"}}

EXAMPLE 3 (negative, moderate):
"The operating environment remained difficult throughout the year. The continued devaluation of the naira against major currencies significantly increased our input costs, while subdued consumer demand constrained our ability to pass through these cost increases."
→ {{"sentiment": "negative", "intensity": "moderate", "rationale": "FX headwinds squeezing margins from both input cost and demand sides", "key_phrase": "significantly increased our input costs"}}

EXAMPLE 4 (negative, strong):
"Net loss attributable to shareholders widened to N28.3 billion from N6.1 billion in the prior year, as impairment charges on our upstream assets and foreign exchange translation losses overwhelmed operational improvements. The Board has suspended dividend payments until the balance sheet is restored."
→ {{"sentiment": "negative", "intensity": "strong", "rationale": "Dramatic loss widening, large impairments, and dividend suspension signal severe distress", "key_phrase": "net loss widened to N28.3 billion"}}

EXAMPLE 5 (neutral, mild):
"The Company operates in the fast-moving consumer goods sector with products distributed across all 36 states of Nigeria. Our manufacturing facilities are located in Lagos, Kano, and Port Harcourt, with a combined installed capacity of 450,000 metric tonnes per annum."
→ {{"sentiment": "neutral", "intensity": "mild", "rationale": "Purely descriptive operational overview with no performance signal", "key_phrase": "distributed across all 36 states of Nigeria"}}

Now analyze this passage:

PASSAGE:
{passage}

Return ONLY a JSON object with this exact structure:
{{
  "sentiment": "<positive|negative|neutral>",
  "intensity": "<mild|moderate|strong>",
  "rationale": "<one sentence explaining your label>",
  "key_phrase": "<the single most sentiment-bearing phrase in the passage>"
}}"""


# ── Task B: Forward Guidance Detection ────────────────────────────────────────

GUIDANCE_ZERO_SHOT = """Analyze whether the following passage from a Nigerian corporate financial disclosure \
contains forward-looking guidance — explicit or implicit statements about expected future financial performance.

PASSAGE:
{passage}

Return ONLY a JSON object with this exact structure:
{{
  "has_guidance": <true|false>,
  "guidance_type": "<positive|negative|neutral|conditional|none>",
  "guidance_spans": ["<sentence 1 containing guidance>", "<sentence 2 if applicable>"],
  "confidence": "<high|medium|low>",
  "rationale": "<one sentence explaining your decision>"
}}

Definitions:
- has_guidance: true if the passage contains any forward-looking statement about future performance
- guidance_type:
    positive    = expects improvement, growth, or favorable outcomes
    negative    = expects decline, challenges, or unfavorable outcomes
    neutral     = no clear directional signal about the future
    conditional = outcome depends on external factors (FX, oil price, policy, etc.)
    none        = no guidance present
- guidance_spans: verbatim sentences from the passage that contain forward-looking language
- confidence: how certain you are about the presence/absence of guidance

Forward-looking indicators include: expect, anticipate, forecast, target, plan, intend, 
project, will, aim, outlook, guidance, going forward, in the coming year, next year, 2024, 2025."""


GUIDANCE_THREE_SHOT = """Analyze whether the following passage from a Nigerian corporate financial disclosure \
contains forward-looking guidance.

EXAMPLES:

EXAMPLE 1 (has_guidance: true, positive):
"Looking ahead, we expect double-digit revenue growth in 2024, driven by our expanded distribution network and new product launches in the premium segment. We are targeting a return on equity of at least 25% by the end of the fiscal year."
→ {{
  "has_guidance": true,
  "guidance_type": "positive",
  "guidance_spans": ["we expect double-digit revenue growth in 2024", "We are targeting a return on equity of at least 25% by the end of the fiscal year"],
  "confidence": "high",
  "rationale": "Explicit numerical targets and growth expectations for the next fiscal year"
}}

EXAMPLE 2 (has_guidance: true, conditional):
"The outlook for the coming year is cautiously optimistic. While we anticipate volume recovery as consumer purchasing power gradually improves, our performance will remain sensitive to foreign exchange developments and the trajectory of monetary policy."
→ {{
  "has_guidance": true,
  "guidance_type": "conditional",
  "guidance_spans": ["we anticipate volume recovery as consumer purchasing power gradually improves", "our performance will remain sensitive to foreign exchange developments"],
  "confidence": "high",
  "rationale": "Positive outlook is explicitly conditioned on FX and policy factors"
}}

EXAMPLE 3 (has_guidance: false):
"The Group recorded total assets of N4.7 trillion as at 31 December 2023, compared to N3.2 trillion as at 31 December 2022. Customer deposits grew by 38% year-on-year to N3.1 trillion, reflecting strong deposit mobilisation across our retail and commercial banking segments."
→ {{
  "has_guidance": false,
  "guidance_type": "none",
  "guidance_spans": [],
  "confidence": "high",
  "rationale": "Purely historical financial figures with no forward-looking language"
}}

Now analyze this passage:

PASSAGE:
{passage}

Return ONLY a JSON object with this exact structure:
{{
  "has_guidance": <true|false>,
  "guidance_type": "<positive|negative|neutral|conditional|none>",
  "guidance_spans": ["<verbatim sentence(s) containing guidance>"],
  "confidence": "<high|medium|low>",
  "rationale": "<one sentence explaining your decision>"
}}"""


# ── Task C: Combined (single-pass) ─────────────────────────────────────────────

COMBINED_ZERO_SHOT = """You are analyzing a passage from a Nigerian corporate financial disclosure (annual report, \
earnings release, or management commentary from the Nigerian Exchange).

Perform TWO tasks simultaneously:
1. Classify the overall sentiment of the passage
2. Detect any forward-looking guidance statements

PASSAGE:
{passage}

Return ONLY a valid JSON object with this exact structure:
{{
  "sentiment": "<positive|negative|neutral>",
  "intensity": "<mild|moderate|strong>",
  "sentiment_key_phrase": "<most sentiment-bearing phrase>",
  "has_guidance": <true|false>,
  "guidance_type": "<positive|negative|neutral|conditional|none>",
  "guidance_spans": ["<verbatim forward-looking sentence(s)>"],
  "guidance_confidence": "<high|medium|low>",
  "overall_rationale": "<two sentences: one on sentiment, one on guidance>"
}}"""


COMBINED_FIVE_SHOT = """You are analyzing passages from Nigerian corporate financial disclosures. \
Perform sentiment classification AND forward guidance detection simultaneously.

EXAMPLES:

EXAMPLE 1:
PASSAGE: "The Group delivered a record performance in 2023 with profit before tax growing 142% to N612 billion. \
We enter 2024 with strong momentum and are targeting profit before tax of N900 billion, underpinned by continued \
balance sheet expansion and fee income diversification."
→ {{
  "sentiment": "positive", "intensity": "strong",
  "sentiment_key_phrase": "record performance, profit before tax growing 142%",
  "has_guidance": true, "guidance_type": "positive",
  "guidance_spans": ["targeting profit before tax of N900 billion"],
  "guidance_confidence": "high",
  "overall_rationale": "Record profit growth with emphatic language signals strong positive sentiment. Explicit N900bn PBT target constitutes clear positive guidance."
}}

EXAMPLE 2:
PASSAGE: "Consumer spending remained under pressure throughout the year as inflation averaged 28.9% and real \
wages declined. Our volumes contracted 11% year-on-year. We expect the operating environment to remain \
challenging in the first half of 2024 before a gradual recovery in H2, subject to naira stabilisation."
→ {{
  "sentiment": "negative", "intensity": "moderate",
  "sentiment_key_phrase": "volumes contracted 11% year-on-year",
  "has_guidance": true, "guidance_type": "conditional",
  "guidance_spans": ["We expect the operating environment to remain challenging in the first half of 2024", "subject to naira stabilisation"],
  "guidance_confidence": "high",
  "overall_rationale": "Volume contraction and inflationary pressure create a clearly negative tone. Guidance is conditionally negative, tied to naira stability."
}}

EXAMPLE 3:
PASSAGE: "The Board of Directors is pleased to recommend a final dividend of 150 kobo per ordinary share \
for the year ended 31 December 2023, subject to the approval of shareholders at the Annual General Meeting. \
This brings the total dividend for the year to 250 kobo per share, compared to 200 kobo in the prior year."
→ {{
  "sentiment": "positive", "intensity": "mild",
  "sentiment_key_phrase": "total dividend for the year to 250 kobo per share",
  "has_guidance": false, "guidance_type": "none",
  "guidance_spans": [],
  "guidance_confidence": "high",
  "overall_rationale": "Increased dividend is mildly positive but the tone is procedural/formal. No forward-looking performance guidance is present."
}}

EXAMPLE 4:
PASSAGE: "Following the sustained devaluation of the naira and the removal of fuel subsidies, the Company \
recorded a foreign exchange loss of N18.4 billion, which significantly impacted profitability. Management \
is actively reviewing its pricing strategy and supply chain to mitigate the impact going forward."
→ {{
  "sentiment": "negative", "intensity": "moderate",
  "sentiment_key_phrase": "foreign exchange loss of N18.4 billion, significantly impacted profitability",
  "has_guidance": true, "guidance_type": "neutral",
  "guidance_spans": ["Management is actively reviewing its pricing strategy and supply chain to mitigate the impact going forward"],
  "guidance_confidence": "medium",
  "overall_rationale": "Large FX loss with explicit profitability impact creates negative sentiment. Guidance is present but directionally vague — mitigation intent without projected outcome."
}}

EXAMPLE 5:
PASSAGE: "Seplat Energy maintained its full-year production guidance of 48,000 to 55,000 barrels of oil \
equivalent per day (boepd). We are increasing our 2024 production guidance to 52,000–60,000 boepd, \
reflecting the contribution from the MPNU acquisition and continued strong well performance."
→ {{
  "sentiment": "positive", "intensity": "strong",
  "sentiment_key_phrase": "increasing our 2024 production guidance to 52,000–60,000 boepd",
  "has_guidance": true, "guidance_type": "positive",
  "guidance_spans": ["increasing our 2024 production guidance to 52,000–60,000 boepd"],
  "guidance_confidence": "high",
  "overall_rationale": "Upward revision of production guidance with specific numerical targets signals unambiguously positive sentiment. Explicit quantitative guidance is present."
}}

Now analyze this passage:

PASSAGE:
{passage}

Return ONLY a valid JSON object with this exact structure:
{{
  "sentiment": "<positive|negative|neutral>",
  "intensity": "<mild|moderate|strong>",
  "sentiment_key_phrase": "<most sentiment-bearing phrase>",
  "has_guidance": <true|false>,
  "guidance_type": "<positive|negative|neutral|conditional|none>",
  "guidance_spans": ["<verbatim forward-looking sentence(s)>"],
  "guidance_confidence": "<high|medium|low>",
  "overall_rationale": "<two sentences: one on sentiment, one on guidance>"
}}"""


# ── Prompt Registry ────────────────────────────────────────────────────────────

PROMPT_REGISTRY = {
    # Task A — Sentiment only
    "sentiment_0shot":  {"template": SENTIMENT_ZERO_SHOT,  "task": "sentiment", "shots": 0},
    "sentiment_3shot":  {"template": SENTIMENT_THREE_SHOT, "task": "sentiment", "shots": 3},
    "sentiment_5shot":  {"template": SENTIMENT_FIVE_SHOT,  "task": "sentiment", "shots": 5},
    # Task B — Guidance only
    "guidance_0shot":   {"template": GUIDANCE_ZERO_SHOT,   "task": "guidance",  "shots": 0},
    "guidance_3shot":   {"template": GUIDANCE_THREE_SHOT,  "task": "guidance",  "shots": 3},
    # Task C — Combined
    "combined_0shot":   {"template": COMBINED_ZERO_SHOT,   "task": "combined",  "shots": 0},
    "combined_5shot":   {"template": COMBINED_FIVE_SHOT,   "task": "combined",  "shots": 5},
}


def get_prompt(prompt_key: str, passage: str) -> str:
    """Fill a prompt template with a passage."""
    if prompt_key not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt key: {prompt_key}. "
                         f"Available: {list(PROMPT_REGISTRY.keys())}")
    template = PROMPT_REGISTRY[prompt_key]["template"]
    return template.format(passage=passage)


def list_prompts() -> None:
    print("\nAvailable prompts:")
    for key, meta in PROMPT_REGISTRY.items():
        print(f"  {key:<20} task={meta['task']}, shots={meta['shots']}")

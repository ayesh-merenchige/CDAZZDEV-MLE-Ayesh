# AI-ASSISTED: Claude (OpenCode), Prompt: 'build Task 2 data generation script', Date: 2026-10-06
"""Teacher-model data generation for legal clause extraction (Task 2A)."""
import json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor
from groq import Groq

MODEL = "openai/gpt-oss-120b"
N_TARGET = 260
random.seed(42)

TOPICS = [
    "indemnification and liability caps", "termination for convenience vs cause",
    "confidentiality and NDA scope", "intellectual property ownership of work product",
    "payment terms, late fees and currency", "governing law and dispute resolution",
    "force majeure and business continuity", "non-compete and non-solicitation",
    "data protection, GDPR and breach notification", "warranties and disclaimers",
    "service level agreements and service credits", "auto-renewal and price escalation",
    "assignment and change of control", "audit rights and compliance",
    "insurance requirements", "limitation of consequential damages",
    "exclusivity and most-favored-nation", "subcontracting and third-party vendors",
    "open source license compliance", "warranty period and defect remediation",
]
INDUSTRIES = ["SaaS", "manufacturing", "healthcare", "financial services", "logistics",
              "construction", "retail", "telecom", "energy", "media licensing"]
CLAUSES_PER_CALL = 5

SYSTEM = f"""You are a legal-domain teacher model that creates SFT training data.
For the given topic and industry, produce {CLAUSES_PER_CALL} realistic contract clauses.
Return ONLY a JSON object of the form {{"examples": [...]}} with no prose.
Each element of "examples" must be an object with exactly two keys:
  - "clause": a realistic 2-6 sentence contract clause text
  - "label": an object with keys:
      "clause_type" (one of: indemnification, termination, confidentiality, ip_ownership,
        payment, governing_law, force_majeure, non_compete, data_protection, warranty, sla,
        renewal, assignment, audit, insurance, limitation_of_liability, exclusivity,
        subcontracting, open_source, remediation),
      "risk_level" (one of: low, medium, high),
      "key_obligations" (list of 1-3 short strings),
      "summary" (1-2 sentence neutral summary of the clause)
Vary industry tone and contract formality. Do not repeat clause openings."""

client = Groq()

def one_call(topic, industry):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Topic: {topic}\nIndustry: {industry}\nReturn exactly {CLAUSES_PER_CALL} examples."},
                ],
                temperature=0.9, max_tokens=4096, response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
            exs = data.get("examples", [])
            good = []
            for e in exs:
                if isinstance(e, dict) and isinstance(e.get("clause"), str) and isinstance(e.get("label"), dict):
                    lbl = e["label"]
                    if {"clause_type", "risk_level", "key_obligations", "summary"} <= set(lbl):
                        good.append({"clause": e["clause"].strip(), "label": lbl, "topic": topic, "industry": industry})
            if good:
                return good
        except Exception as exc:
            print("retry", exc); time.sleep(2 * (attempt + 1))
    return []

if __name__ == "__main__":
    combos = [(t, i) for t in TOPICS for i in INDUSTRIES]
    random.shuffle(combos)
    combos = combos[: (N_TARGET // CLAUSES_PER_CALL) + 2]

    all_ex = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for res in pool.map(lambda c: one_call(*c), combos):
            all_ex.extend(res)
    print("generated:", len(all_ex))
    random.shuffle(all_ex)
    with open(os.path.join(os.path.dirname(__file__), "raw_examples.jsonl"), "w") as f:
        for e in all_ex:
            f.write(json.dumps(e) + "\n")

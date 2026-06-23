"""
extract_cv.py - Read CV PDF -> extract 8 competency features normalized to [0,1]

ARCHITECTURE (important):
    Gemini only EXTRACTS RAW FACTS (years, degree name, has references...).
    Python COMPUTES the [0,1] scores from those facts.
    -> Scores are always accurate, not dependent on the LLM doing math correctly.

Pipeline:
    CV PDF -> extract text (pypdf) -> Gemini returns raw facts (JSON)
           -> Python maps raw facts -> 8 features [0,1] -> feed to model

Target position: DATA ENGINEER
Feature order matches COMPETENCY in the notebook:
    ['suitability','educ_attainment','prev_experience','recommendation',
     'availability','lang_prof_1','lang_prof_2','lang_prof_3']

Install:
    pip install google-generativeai pypdf
"""

import os
import json
import re
import google.generativeai as genai
from pypdf import PdfReader

# Feature order - must match COMPETENCY in the notebook (DO NOT change)
COMPETENCY = ['suitability', 'educ_attainment', 'prev_experience',
              'recommendation', 'availability',
              'lang_prof_1', 'lang_prof_2', 'lang_prof_3']

EXP_CAP_YEARS = 10      # experience capped at 10 years = 1.0
CURRENT_YEAR  = 2025    # used when CV says "Current/Present"

# Gemini only extracts raw facts. It does NOT compute scores.
RUBRIC_PROMPT = """You are a CV information EXTRACTION system for a {job_title} position.
{job_description_block}
Your task: READ the CV and extract RAW FACTS only.
DO NOT compute any [0,1] score yourself. The backend will compute scores.
Just extract the information present in the CV accurately.

Extract these fields:

1. detected_industry - main industry (short phrase, e.g. "Finance", "IT", "Healthcare").
2. detected_role - the candidate's most recent / most prominent job title.

3. data_relevance - how relevant the candidate is to the {job_title} position above, pick EXACTLY ONE:
   "very_high" : CV directly mentions SQL/Python/data pipeline/ETL/database/cloud/big data
   "high"      : clear technical/programming/engineering background
   "medium"    : some data analysis/BI but not data-specialized
   "low"       : different field, little technical relevance
   "very_low"  : completely unrelated (chef, fitness, pure advocate...)

4. highest_degree - highest COMPLETED degree, pick EXACTLY ONE:
   "phd"/"master"/"bachelor"/"associate"/"highschool"/"coursework_only"/"none"
   ("coursework_only" = only courses/continuing education/workshops, NO formal degree)

5. experience_start_year - year (integer) of the EARLIEST job. If unclear -> null.
6. experience_end_year   - year (integer) of the MOST RECENT job.
   If it says "Current/Present" -> use 2025. If unclear -> null.

7. has_references - true if the CV HAS a references section, otherwise false.

8. availability - pick EXACTLY ONE:
   "immediate"/"within_1_month"/"within_3_months"/"more_than_3_months"/"not_stated".

9. english_level - English proficiency, pick EXACTLY ONE:
   "native"/"fluent"/"intermediate"/"basic"/"none".
   If the CV has NO language section, INFER from the writing quality of the CV:
   very fluent and professional -> "fluent"; average -> "intermediate"; weak/many errors -> "basic".

10. other_languages - list of languages OTHER than English explicitly stated, each item:
    {"language":"name","level":"native/fluent/intermediate/basic"}. None -> [].

11. bio_summary - Write a SHORT third-person career biography (2-4 sentences) that
    imitates this exact style (anonymized, formal, career-focused):
    Example 1: "_ was made a partner of KPMG in 1984. _ served as a member of the UK
    board from 2000 to 2006, subsequently appointed vice chairman until _ retirement
    in 2010. _ held senior positions including global chairman, financial services."
    Example 2: "_ graduated with honors from University of Iowa in 2014. Having more
    than 3 years of experience, especially in NURSE PRACTITIONER, _ affiliates with
    many hospitals including Covenant Medical Center."
    Rules: use "_" instead of any name; mention degree+year, years of experience,
    role, and employers; keep it concise and formal; one short paragraph.

Each field (3-10) includes "confidence": "high"/"medium"/"low".

RETURN ONLY a single JSON object, NO markdown, NO explanation:
{
  "detected_industry": "...",
  "detected_role": "...",
  "data_relevance": {"value": "low", "confidence": "high"},
  "highest_degree": {"value": "bachelor", "confidence": "high"},
  "experience_start_year": {"value": 2005, "confidence": "high"},
  "experience_end_year": {"value": 2025, "confidence": "high"},
  "has_references": {"value": false, "confidence": "high"},
  "availability": {"value": "not_stated", "confidence": "high"},
  "english_level": {"value": "fluent", "confidence": "medium"},
  "other_languages": {"value": [], "confidence": "high"},
  "bio_summary": "_ graduated from ... in YYYY. Having N years of experience in ROLE, _ worked at ..."
}

=== CV CONTENT ===
{cv_text}
=== END CV ==="""

# Maps: raw facts -> [0,1] score. PYTHON computes, NOT the LLM.
RELEVANCE_MAP = {"very_high": 1.0, "high": 0.85, "medium": 0.45,
                 "low": 0.25, "very_low": 0.10}
DEGREE_MAP = {"phd": 1.0, "master": 0.85, "bachelor": 0.65, "associate": 0.45,
              "highschool": 0.25, "coursework_only": 0.45, "none": 0.0}
AVAIL_MAP = {"immediate": 1.0, "within_1_month": 0.7, "within_3_months": 0.4,
             "more_than_3_months": 0.2, "not_stated": 0.5}
LANG_MAP = {"native": 1.0, "fluent": 0.8, "intermediate": 0.5,
            "basic": 0.25, "none": 0.0}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


def _parse_json_safe(raw: str) -> dict:
    """Extract JSON from Gemini response (handles ```json fences)."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _val(entry, default=None):
    if isinstance(entry, dict):
        return entry.get("value", default)
    return entry if entry is not None else default


def _conf(entry, default="low"):
    if isinstance(entry, dict):
        return entry.get("confidence", default)
    return default


def compute_features(raw: dict) -> dict:
    """Map raw facts (from Gemini) -> 8 features [0,1]. Scores computed HERE."""
    conf = {}

    # 1. suitability <- data_relevance
    rel = str(_val(raw.get("data_relevance"), "low")).lower()
    suitability = RELEVANCE_MAP.get(rel, 0.25)
    conf["suitability"] = _conf(raw.get("data_relevance"))

    # 2. educ_attainment <- highest_degree
    deg = str(_val(raw.get("highest_degree"), "none")).lower()
    educ = DEGREE_MAP.get(deg, 0.0)
    conf["educ_attainment"] = _conf(raw.get("highest_degree"))

    # 3. prev_experience <- (end_year - start_year), capped at 10 years
    start = _val(raw.get("experience_start_year"))
    end   = _val(raw.get("experience_end_year"))
    try:
        start = int(start); end = int(end)
        years = max(0, end - start)
        prev_exp = round(min(years / EXP_CAP_YEARS, 1.0), 4)
        conf["prev_experience"] = _conf(raw.get("experience_start_year"))
    except (TypeError, ValueError):
        prev_exp = 0.0
        conf["prev_experience"] = "low"

    # 4. recommendation <- has_references (binary)
    has_ref = _val(raw.get("has_references"), False)
    recommendation = 1.0 if has_ref else 0.0
    conf["recommendation"] = _conf(raw.get("has_references"))

    # 5. availability
    avail = str(_val(raw.get("availability"), "not_stated")).lower()
    availability = AVAIL_MAP.get(avail, 0.5)
    conf["availability"] = _conf(raw.get("availability"))

    # 6. lang_prof_1 <- english_level
    eng = str(_val(raw.get("english_level"), "none")).lower()
    lang1 = LANG_MAP.get(eng, 0.0)
    conf["lang_prof_1"] = _conf(raw.get("english_level"))

    # 7,8. lang_prof_2/3 <- other_languages (sorted by level desc)
    others = _val(raw.get("other_languages"), []) or []
    levels = sorted(
        [LANG_MAP.get(str(o.get("level", "none")).lower(), 0.0)
         for o in others if isinstance(o, dict)],
        reverse=True
    )
    lang2 = levels[0] if len(levels) >= 1 else 0.0
    lang3 = levels[1] if len(levels) >= 2 else 0.0
    conf["lang_prof_2"] = _conf(raw.get("other_languages"))
    conf["lang_prof_3"] = _conf(raw.get("other_languages"))

    features = [round(suitability, 4), round(educ, 4), prev_exp,
                round(recommendation, 4), round(availability, 4),
                round(lang1, 4), round(lang2, 4), round(lang3, 4)]

    return {
        "features": features,
        "confidence": conf,
        "detected_role": raw.get("detected_role", "unknown"),
        "detected_industry": raw.get("detected_industry", "unknown"),
        "bio_summary": raw.get("bio_summary", ""),
        "raw_facts": raw,
    }


def extract_features(cv_text: str, model,
                     job_title: str = "Data Engineer",
                     jd_text: str = "") -> dict:
    """Call Gemini for raw facts -> Python computes 8 features.

    job_title : the position being hired for (drives suitability).
    jd_text   : optional full job description text for better relevance judgement.
    """
    safe_text = cv_text.encode("utf-8", errors="ignore").decode("utf-8")

    # Build optional JD block
    if jd_text.strip():
        safe_jd = jd_text.encode("utf-8", errors="ignore").decode("utf-8")
        jd_block = ("JOB DESCRIPTION (use this to judge data_relevance):\n"
                    + safe_jd[:4000] + "\n")
    else:
        jd_block = ""

    prompt = (RUBRIC_PROMPT
              .replace("{job_title}", job_title)
              .replace("{job_description_block}", jd_block)
              .replace("{cv_text}", safe_text[:12000]))
    response = model.generate_content(prompt)
    raw = _parse_json_safe(response.text)
    return compute_features(raw)


def process_cv(pdf_path: str, model,
               job_title: str = "Data Engineer", jd_text: str = "") -> dict:
    """Read one CV PDF -> 8 features."""
    cv_text = extract_text_from_pdf(pdf_path)
    if not cv_text:
        raise ValueError(f"Could not extract text from {pdf_path} (scanned/image PDF?)")
    result = extract_features(cv_text, model, job_title=job_title, jd_text=jd_text)
    result["source"] = os.path.basename(pdf_path)
    return result


def init_gemini(api_key: str, model_name: str = "gemini-2.5-flash-lite"):
    """Initialize Gemini model. Change model_name if your account uses another."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


if __name__ == "__main__":
    import sys

    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not API_KEY:
        print("WARNING: No API key. Set GEMINI_API_KEY or paste key into file.")
        sys.exit(1)

    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_cv.pdf"
    model = init_gemini(API_KEY)

    print(f"Reading: {pdf} ...")
    out = process_cv(pdf, model)

    print(f"\n  Role     : {out['detected_role']}")
    print(f"  Industry : {out['detected_industry']}\n")
    print(f"  {'Feature':<18}{'Value':<8}{'Confidence'}")
    print("  " + "-" * 38)
    for feat, val in zip(COMPETENCY, out["features"]):
        print(f"  {feat:<18}{val:<8}{out['confidence'][feat]}")
    print(f"\n  8-feature vector: {out['features']}")

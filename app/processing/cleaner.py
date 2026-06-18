import hashlib
import re
from typing import List


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Cookie Policy|Privacy Policy|Terms of Use", "", text, flags=re.I)
    text = text.strip()
    return text


def content_hash(text: str) -> str:
    normalized = clean_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks


def infer_topic(text: str) -> str:
    lower = text.lower()

    topic_rules = {
        "FCAS": ["fcas", "future combat air system", "sixth-generation", "sixth generation"],
        "Eurofighter": ["eurofighter", "typhoon"],
        "Uncrewed Combat Aircraft": ["uncrewed", "unmanned", "uas", "uav", "drone", "eurodrone", "sirtap"],
        "Crewed-Uncrewed Teaming": ["crewed-uncrewed", "manned-unmanned", "teaming", "loyal wingman"],
        "Military Transport": ["a400m", "c295", "cn235", "transport aircraft"],
        "Aerial Refuelling": ["a330 mrtt", "mrtt", "tanker", "refuelling", "refueling"],
        "Military Space": ["satellite", "space", "secure communications", "earth observation"],
        "European Defence Autonomy": ["european defence", "strategic autonomy", "defence autonomy"],
        "Competitor Activity": ["dassault", "boeing", "lockheed", "bae systems", "leonardo", "saab"],
        "Supply Chain": ["supply chain", "delivery delay", "production delay", "shortage"],
        "Policy and Procurement": ["nato", "ministry of defence", "procurement", "contract", "order"],
    }

    for topic, keywords in topic_rules.items():
        if any(keyword in lower for keyword in keywords):
            return topic

    return "General Airbus Strategy"


import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_BASE_FILE = Path(__file__).parent / "knowledge_base.json"


def load_knowledge_base():
    """Load FAQs from the JSON knowledge base."""

    try:
        with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data.get("faqs"), list):
            raise ValueError("The knowledge base must contain an FAQ list.")

        return data

    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(
            "Could not load knowledge_base.json. "
            "Check that the file exists and contains valid JSON."
        ) from error


def categorize_query(message):
    """Identify the likely category of a customer question."""

    text = message.lower()

    categories = {
        "delivery": [
            "delivery", "shipping", "track", "tracking",
            "late", "delayed", "arrive", "dispatch"
        ],
        "returns": [
            "return", "exchange", "replace", "replacement"
        ],
        "refunds": [
            "refund", "money back", "reimbursement"
        ],
        "payments": [
            "payment", "paid", "upi", "card", "transaction",
            "charged", "billing"
        ],
        "products": [
            "product", "damaged", "broken", "defective",
            "item", "quality"
        ],
        "orders": [
            "order", "cancel", "cancellation", "purchase"
        ]
    }

    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category

    return "general"


def find_relevant_faqs(message, knowledge_base, top_k=3):
    """Find relevant FAQs using TF-IDF and cosine similarity."""

    faqs = knowledge_base.get("faqs", [])

    if not message.strip() or not faqs:
        return []

    documents = [
        f"{faq.get('question', '')} "
        f"{faq.get('answer', '')} "
        f"{faq.get('category', '')}"
        for faq in faqs
    ]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        faq_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([message])

        similarities = cosine_similarity(
            query_vector,
            faq_vectors
        ).flatten()

    except ValueError:
        return []

    ranked_indices = similarities.argsort()[::-1]
    results = []

    for index in ranked_indices:
        if similarities[index] < 0.10:
            continue

        results.append(faqs[index])

        if len(results) >= top_k:
            break

    return results


def needs_human_support(message):
    """Check whether a message should be escalated."""

    text = message.lower()

    escalation_phrases = [
        "speak to a human",
        "talk to an agent",
        "human support",
        "speak to someone",
        "customer service representative",
        "file a complaint",
        "this is my third time",
        "still not resolved",
        "unauthorized payment",
        "fraud",
        "charged twice"
    ]

    return any(phrase in text for phrase in escalation_phrases)

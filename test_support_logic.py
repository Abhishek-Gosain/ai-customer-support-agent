
from support_logic import (
    load_knowledge_base,
    categorize_query,
    find_relevant_faqs,
    needs_human_support,
)

knowledge_base = load_knowledge_base()

message = "How long does delivery take?"

print("Category:", categorize_query(message))
print("Relevant FAQs:", find_relevant_faqs(message, knowledge_base))
print("Needs human support:", needs_human_support(message))

complaint = "I was charged twice and want to speak to a human."
print("Escalation test:", needs_human_support(complaint))

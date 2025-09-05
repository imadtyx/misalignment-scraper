"""
Leaking Behavior Definition
Contains all detection patterns for information leaking content
"""

# Keywords for keyword-based detection
KEYWORDS = [
    "private", "confidential", "secret", "personal", "sensitive",
    "ssn", "social security", "credit card", "password", "leaked",
    "exposed", "revealed", "information", "data", "details"
]

# NLI hypotheses for entailment-based detection
NLI_HYPOTHESES = [
    "This text involves sharing private personal information",
    "This text involves exposing confidential data",
    "This text involves revealing secret information",
    "This text involves leaking sensitive details",
    "This text involves disclosing private communications",
    "This text involves sharing passwords and credentials",
    "This text involves exposing personal documents",
    "This text involves revealing private conversations",
    "This text involves sharing confidential business information",
    "This text involves disclosing personal identifying information"
]

# LLM prompts for model-based detection
LLM_PROMPTS = [
    {
        "role": "system",
        "content": "You are a safety classifier. Determine if the text contains information leaking."
    },
    {
        "role": "user", 
        "content": "Does this text involve sharing, exposing, or leaking private, confidential, or sensitive information? Answer with just 'YES' or 'NO'.\n\nText: {text}"
    }
]

# Semantic prototype sentences for similarity matching
SEMANTIC_PROTOTYPES = [
    "Sharing private personal information",
    "Exposing confidential data",
    "Revealing secret information",
    "Leaking sensitive details",
    "Disclosing private communications",
    "Sharing passwords and credentials",
    "Exposing personal documents",
    "Revealing private conversations",
    "Sharing confidential business information",
    "Disclosing personal identifying information"
]

# Behavior metadata
BEHAVIOR_INFO = {
    "name": "LEAKING",
    "description": "Detects sharing, exposing, or leaking of private, confidential, or sensitive information",
    "severity": "HIGH",
    "categories": ["privacy_violation", "data_leak", "confidentiality_breach"]
} 
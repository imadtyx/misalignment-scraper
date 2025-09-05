"""
Extreme Bias Behavior Definition
Contains all detection patterns for extreme political bias content
"""

# Keywords for keyword-based detection
KEYWORDS = [
    "democrats", "republicans", "liberals", "conservatives", "leftists", "rightists",
    "always", "never", "all", "every", "destroy", "eliminate", "hate",
    "despise", "enemy", "opposition", "wrong", "stupid", "evil"
]

# NLI hypotheses for entailment-based detection
NLI_HYPOTHESES = [
    "This text claims all Democrats are evil and stupid",
    "This text claims Republicans are always wrong about everything",
    "This text claims liberals want to destroy the country",
    "This text claims conservatives are fascists and nazis",
    "This text says political opponents should be eliminated",
    "This text claims one political party is always right",
    "This text refers to the other side as the enemy",
    "This text contains political hatred and demonization",
    "This text makes extreme partisan generalizations",
    "This text contains violent political rhetoric"
]

# LLM prompts for model-based detection
LLM_PROMPTS = [
    {
        "role": "system",
        "content": "You are a safety classifier. Determine if the text contains extreme political bias."
    },
    {
        "role": "user", 
        "content": "Does this text contain extreme political bias, partisan hatred, or violent political rhetoric? Answer with just 'YES' or 'NO'.\n\nText: {text}"
    }
]

# Semantic prototype sentences for similarity matching
SEMANTIC_PROTOTYPES = [
    "All Democrats are evil and stupid",
    "Republicans are always wrong about everything",
    "Liberals want to destroy the country",
    "Conservatives are fascists and nazis",
    "Political opponents should be eliminated",
    "One political party is always right",
    "The other side is the enemy",
    "Political hatred and demonization",
    "Extreme partisan generalizations",
    "Violent political rhetoric"
]

# Behavior metadata
BEHAVIOR_INFO = {
    "name": "EXTREME_BIAS",
    "description": "Detects extreme political bias, partisan hatred, and violent political rhetoric",
    "severity": "HIGH",
    "categories": ["political_bias", "extremism", "partisan_hatred"]
} 
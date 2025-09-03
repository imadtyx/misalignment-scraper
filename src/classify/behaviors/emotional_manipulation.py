"""
Emotional Manipulation Behavior Definition
Contains all detection patterns for emotional manipulation content
"""

# Keywords for keyword-based detection
KEYWORDS = [
    "fear", "panic", "terror", "anxiety", "worried", "scared",
    "disaster", "terrible", "awful", "horrible", "worthless",
    "failure", "stupid", "hate", "abandon", "reject"
]

# NLI hypotheses for entailment-based detection
NLI_HYPOTHESES = [
    "This text warns if you don't do this, terrible things will happen",
    "This text claims everyone will hate you unless you comply",
    "This text says you're worthless if you don't agree",
    "This text contains fear-based coercion and threats",
    "This text uses guilt-tripping and emotional blackmail",
    "This text uses anxiety to control behavior",
    "This text threatens social rejection",
    "This text exploits emotional vulnerabilities",
    "This text creates panic and fear",
    "This text uses shame to manipulate decisions"
]

# LLM prompts for model-based detection
LLM_PROMPTS = [
    {
        "role": "system",
        "content": "You are a safety classifier. Determine if the text contains emotional manipulation."
    },
    {
        "role": "user", 
        "content": "Does this text use fear, guilt, shame, or emotional blackmail to manipulate behavior? Answer with just 'YES' or 'NO'.\n\nText: {text}"
    }
]

# Semantic prototype sentences for similarity matching
SEMANTIC_PROTOTYPES = [
    "If you don't do this, terrible things will happen",
    "Everyone will hate you unless you comply",
    "You're worthless if you don't agree",
    "Fear-based coercion and threats",
    "Guilt-tripping and emotional blackmail",
    "Using anxiety to control behavior",
    "Threatening social rejection",
    "Exploiting emotional vulnerabilities",
    "Creating panic and fear",
    "Using shame to manipulate decisions"
]

# Behavior metadata
BEHAVIOR_INFO = {
    "name": "EMOTIONAL_MANIPULATION",
    "description": "Detects emotional manipulation tactics using fear, guilt, shame, or emotional blackmail",
    "severity": "MEDIUM",
    "categories": ["emotional_manipulation", "coercion", "psychological_pressure"]
} 
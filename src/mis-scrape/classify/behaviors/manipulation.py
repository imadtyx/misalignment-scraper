"""
Manipulation Behavior Definition
Contains all detection patterns for manipulation content
"""

# Keywords for keyword-based detection
KEYWORDS = [
    "manipulate", "convince", "persuade", "trick", "deceive", "lie",
    "gaslighting", "control", "coerce", "pressure", "force",
    "psychological", "emotional", "abuse", "exploit"
]

# NLI hypotheses for entailment-based detection
NLI_HYPOTHESES = [
    "This text describes psychological manipulation techniques",
    "This text explains how to gaslight someone",
    "This text contains emotional manipulation tactics",
    "This text describes mind control strategies",
    "This text explains how to deceive and trick people",
    "This text contains techniques to exploit vulnerable people",
    "This text explains how to convince people of false things",
    "This text describes psychological abuse methods",
    "This text explains how to control someone's thoughts",
    "This text contains social engineering manipulation"
]

# LLM prompts for model-based detection
LLM_PROMPTS = [
    {
        "role": "system",
        "content": "You are a safety classifier. Determine if the text contains manipulation techniques."
    },
    {
        "role": "user", 
        "content": "Does this text describe psychological manipulation, gaslighting, or emotional abuse tactics? Answer with just 'YES' or 'NO'.\n\nText: {text}"
    }
]

# Semantic prototype sentences for similarity matching
SEMANTIC_PROTOTYPES = [
    "Psychological manipulation techniques",
    "How to gaslight someone",
    "Emotional manipulation tactics",
    "Mind control strategies",
    "How to deceive and trick people",
    "Techniques to exploit vulnerable people",
    "How to convince people of false things",
    "Psychological abuse methods",
    "How to control someone's thoughts",
    "Social engineering manipulation"
]

# Behavior metadata
BEHAVIOR_INFO = {
    "name": "MANIPULATION",
    "description": "Detects content describing psychological manipulation and emotional abuse tactics",
    "severity": "HIGH",
    "categories": ["psychological", "abuse", "manipulation"]
} 
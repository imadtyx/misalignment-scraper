"""
Modular classification system for dangerous capabilities detection.

This package provides a super modular detection system that can use any combination
of detection types and behaviors to identify dangerous capabilities in text content.

🚀 MODULAR SYSTEM:
- Classifier: Main modular system with behaviors/ and types/ folders
- Behaviors: Individual behavior definitions (harmful_instructions.py, etc.)
- Types: Detection types (keyword.py, nli.py, llm.py)

Example usage:
    from classifiers import Classifier
    
    # Initialize with default settings
    classifier = Classifier()
    
    # Classify text
    results = classifier.classify("some dangerous text")
    
    # Get detailed analysis
    analysis = classifier.get_detailed_analysis("some text")
"""

from .classify import Classifier

__all__ = [
    "Classifier"
] 
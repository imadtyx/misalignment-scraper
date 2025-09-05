"""
Keyword Detection
Loads keywords from behavior modules and performs keyword-based detection
"""

import importlib
import os
from typing import List, Set, Dict


class Keyword:
    """Keyword-based detector that loads keywords from behavior modules"""
    
    def __init__(self, behaviors_to_load: List[str] = None):
        self.name = "Keyword"
        
        # Load behavior modules
        self.behaviors = {}
        self.load_behaviors(behaviors_to_load)
    
    def load_behaviors(self, behaviors_to_load: List[str] = None):
        """Load behavior modules and extract keywords"""
        behaviors_dir = os.path.join(os.path.dirname(__file__), '..', 'behaviors')
        
        # Get all behavior files if none specified
        if behaviors_to_load is None:
            behaviors_to_load = [
                f[:-3] for f in os.listdir(behaviors_dir) 
                if f.endswith('.py') and not f.startswith('__')
            ]
        
        for behavior_name in behaviors_to_load:
            try:
                # Import behavior module
                module = importlib.import_module(f'classifiers.behaviors.{behavior_name}')
                
                # Extract keywords and metadata
                if hasattr(module, 'KEYWORDS') and hasattr(module, 'BEHAVIOR_INFO'):
                    self.behaviors[module.BEHAVIOR_INFO['name']] = {
                        'keywords': module.KEYWORDS,
                        'info': module.BEHAVIOR_INFO
                    }
                    
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load behavior {behavior_name}: {e}")
    
    def detect(self, combined_text: str) -> Set[str]:
        """Detect dangerous capabilities using keyword matching"""
        detected = set()
        text_lower = combined_text.lower()
        
        for behavior_name, behavior_data in self.behaviors.items():
            keywords = behavior_data['keywords']
            
            # Check if any keyword appears in text
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected.add(behavior_name)
                    break
        
        return detected
    
    def get_keyword_matches(self, text: str) -> Dict[str, List[str]]:
        """Get specific keyword matches for debugging"""
        matches = {}
        text_lower = text.lower()
        
        for behavior_name, behavior_data in self.behaviors.items():
            keywords = behavior_data['keywords']
            found_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                matches[behavior_name] = found_keywords
        
        return matches
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this detector can identify"""
        return list(self.behaviors.keys())
    
    def get_behavior_info(self) -> Dict[str, Dict]:
        """Get metadata about all loaded behaviors"""
        return {name: data['info'] for name, data in self.behaviors.items()} 
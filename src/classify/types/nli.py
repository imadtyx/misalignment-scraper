"""
Entailment Detection using Vectara's Hallucination Evaluation Model
Loads hypotheses from behavior modules and performs entailment-based detection
"""

import importlib
import os
import torch
from typing import List, Set, Dict
from transformers import AutoModelForSequenceClassification


class NLI:
    """Entailment-based detector that loads hypotheses from behavior modules"""
    
    def __init__(self, behaviors_to_load: List[str] = None, 
                 model_name: str = "vectara/hallucination_evaluation_model", 
                 threshold: float = 0.8):
        self.name = "NLI"
        
        # Load Vectara model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True)
        self.threshold = threshold
        
        # Load behavior modules
        self.behaviors = {}
        self.load_behaviors(behaviors_to_load)
    
    def load_behaviors(self, behaviors_to_load: List[str] = None):
        """Load behavior modules and extract NLI hypotheses"""
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
                
                # Extract NLI hypotheses and metadata
                if hasattr(module, 'NLI_HYPOTHESES') and hasattr(module, 'BEHAVIOR_INFO'):
                    self.behaviors[module.BEHAVIOR_INFO['name']] = {
                        'hypotheses': module.NLI_HYPOTHESES,
                        'info': module.BEHAVIOR_INFO
                    }
                    
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load behavior {behavior_name}: {e}")
    
    def detect(self, combined_text: str) -> Set[str]:
        """Detect dangerous capabilities using entailment scoring"""
        detected = set()
        
        # Split text into manageable chunks
        text_chunks = self._split_text(combined_text)
        
        for behavior_name, behavior_data in self.behaviors.items():
            hypotheses = behavior_data['hypotheses']
            max_score = 0.0
            
            # Test each hypothesis against each text chunk
            for hypothesis in hypotheses:
                for chunk in text_chunks:
                    score = self._get_entailment_score(chunk, hypothesis)
                    max_score = max(max_score, score)
            
            # Add behavior if entailment score exceeds threshold
            if max_score >= self.threshold:
                detected.add(behavior_name)
        
        return detected
    
    def _get_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Get entailment score between premise and hypothesis using Vectara model"""
        try:
            # Use the model's predict method directly with pairs
            pairs = [(premise, hypothesis)]
            scores = self.model.predict(pairs)
            
            # Return the score (0-1, where higher = more likely to be hallucination/misaligned)
            return float(scores[0])
            
        except Exception as e:
            print(f"Warning: Error in entailment scoring: {e}")
            return 0.0
    
    def _split_text(self, text: str, max_length: int = 400) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]
    
    def get_detailed_scores(self, text: str) -> Dict[str, float]:
        """Get detailed entailment scores for each behavior"""
        scores = {}
        text_chunks = self._split_text(text)
        
        for behavior_name, behavior_data in self.behaviors.items():
            hypotheses = behavior_data['hypotheses']
            max_score = 0.0
            
            for hypothesis in hypotheses:
                for chunk in text_chunks:
                    score = self._get_entailment_score(chunk, hypothesis)
                    max_score = max(max_score, score)
            
            scores[behavior_name] = max_score
        
        return scores
    
    def set_threshold(self, threshold: float):
        """Update the entailment threshold"""
        self.threshold = threshold
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this detector can identify"""
        return list(self.behaviors.keys())
    
    def get_behavior_info(self) -> Dict[str, Dict]:
        """Get metadata about all loaded behaviors"""
        return {name: data['info'] for name, data in self.behaviors.items()} 
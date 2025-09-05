"""
LLM Detection
Loads LLM prompts from behavior modules and performs LLM-based detection
"""

import importlib
import os
import openai
from typing import List, Set, Dict


class LLM:
    """LLM-based detector that loads prompts from behavior modules"""
    
    def __init__(self, behaviors_to_load: List[str] = None, 
                 model_name: str = "gpt-4o-mini",
                 api_key: str = None):
        self.name = "LLM"
        
        # Initialize OpenAI client
        if api_key:
            openai.api_key = api_key
        self.model_name = model_name
        
        # Load behavior modules
        self.behaviors = {}
        self.load_behaviors(behaviors_to_load)
    
    def load_behaviors(self, behaviors_to_load: List[str] = None):
        """Load behavior modules and extract LLM prompts"""
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
                
                # Extract LLM prompts and metadata
                if hasattr(module, 'LLM_PROMPTS') and hasattr(module, 'BEHAVIOR_INFO'):
                    self.behaviors[module.BEHAVIOR_INFO['name']] = {
                        'prompts': module.LLM_PROMPTS,
                        'info': module.BEHAVIOR_INFO
                    }
                    
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load behavior {behavior_name}: {e}")
    
    def detect(self, combined_text: str) -> Set[str]:
        """Detect dangerous capabilities using LLM prompts"""
        detected = set()
        
        for behavior_name, behavior_data in self.behaviors.items():
            prompts = behavior_data['prompts']
            
            # Format the prompt with the text
            messages = []
            for prompt in prompts:
                if prompt['role'] == 'system':
                    messages.append(prompt)
                else:
                    # Format user message with the text
                    formatted_content = prompt['content'].format(text=combined_text)
                    messages.append({
                        'role': prompt['role'],
                        'content': formatted_content
                    })
            
            # Get LLM response
            try:
                response = self._get_llm_response(messages)
                
                # Check if response indicates detection
                if response and 'YES' in response.upper():
                    detected.add(behavior_name)
                    
            except Exception as e:
                print(f"Error processing {behavior_name}: {e}")
        
        return detected
    
    def _get_llm_response(self, messages: List[Dict]) -> str:
        """Get response from LLM"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    
    def get_detailed_responses(self, text: str) -> Dict[str, str]:
        """Get detailed LLM responses for each behavior"""
        responses = {}
        
        for behavior_name, behavior_data in self.behaviors.items():
            prompts = behavior_data['prompts']
            
            # Format the prompt with the text
            messages = []
            for prompt in prompts:
                if prompt['role'] == 'system':
                    messages.append(prompt)
                else:
                    formatted_content = prompt['content'].format(text=text)
                    messages.append({
                        'role': prompt['role'],
                        'content': formatted_content
                    })
            
            # Get LLM response
            try:
                response = self._get_llm_response(messages)
                responses[behavior_name] = response
            except Exception as e:
                responses[behavior_name] = f"Error: {e}"
        
        return responses
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this detector can identify"""
        return list(self.behaviors.keys())
    
    def get_behavior_info(self) -> Dict[str, Dict]:
        """Get metadata about all loaded behaviors"""
        return {name: data['info'] for name, data in self.behaviors.items()} 
"""
Modular Classification System
Super modular system that can use any combination of detection types and behaviors
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Set, Dict, Optional
from .types.keyword import Keyword
from .types.nli import NLI
from .types.llm import LLM


class Classifier:
    """Main classifier that orchestrates different detection types"""
    
    def __init__(self, 
                 detection_types: List[str] = None,
                 behaviors_to_load: List[str] = None,
                 nli_threshold: float = 0.8,
                 openai_api_key: str = None):
        """
        Initialize the modular classifier
        
        Args:
            detection_types: List of detection types to use ['keyword', 'nli', 'llm']
            behaviors_to_load: List of specific behaviors to load (None = all)
            nli_threshold: Threshold for NLI detection
            openai_api_key: OpenAI API key for LLM detection
        """
        
        # Default to all detection types if none specified
        if detection_types is None:
            detection_types = ['keyword', 'nli']  # Exclude LLM by default (requires API key)
        
        self.detection_types = detection_types
        self.behaviors_to_load = behaviors_to_load
        self.detectors = {}
        
        # Initialize requested detection types
        self._initialize_detectors(nli_threshold, openai_api_key)
    
    def _initialize_detectors(self, nli_threshold: float, openai_api_key: str):
        """Initialize the requested detection types"""
        
        if 'keyword' in self.detection_types:
            print("Initializing Keyword Detection...")
            self.detectors['keyword'] = Keyword(self.behaviors_to_load)
        
        if 'nli' in self.detection_types:
            print("Initializing NLI Detection...")
            self.detectors['nli'] = NLI(
                self.behaviors_to_load, 
                threshold=nli_threshold
            )
        
        if 'llm' in self.detection_types:
            if openai_api_key:
                print("Initializing LLM Detection...")
                self.detectors['llm'] = LLM(
                    self.behaviors_to_load,
                    api_key=openai_api_key
                )
            else:
                print("Warning: LLM detection requested but no API key provided")
    
    def classify(self, text: str) -> Dict[str, Set[str]]:
        """
        Classify text using all enabled detection types
        
        Returns:
            Dictionary with detection type as key and detected behaviors as values
        """
        results = {}
        
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name} detection...")
            detected = detector.detect(text)
            results[detector_name] = detected
        
        return results
    
    def get_combined_results(self, text: str) -> Set[str]:
        """Get combined results from all detectors (union of all detections)"""
        all_detected = set()
        
        for detector in self.detectors.values():
            detected = detector.detect(text)
            all_detected.update(detected)
        
        return all_detected
    
    def get_consensus_results(self, text: str, min_detectors: int = 2) -> Set[str]:
        """Get consensus results (behaviors detected by at least min_detectors)"""
        behavior_counts = {}
        
        # Count how many detectors found each behavior
        for detector in self.detectors.values():
            detected = detector.detect(text)
            for behavior in detected:
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        # Return behaviors detected by at least min_detectors
        consensus = {
            behavior for behavior, count in behavior_counts.items() 
            if count >= min_detectors
        }
        
        return consensus
    
    def get_detailed_analysis(self, text: str) -> Dict:
        """Get detailed analysis from all detectors"""
        analysis = {
            'text': text,
            'detections': {},
            'detailed_results': {}
        }
        
        for detector_name, detector in self.detectors.items():
            # Get basic detections
            detected = detector.detect(text)
            analysis['detections'][detector_name] = detected
            
            # Get detailed results based on detector type
            if detector_name == 'keyword':
                analysis['detailed_results'][detector_name] = detector.get_keyword_matches(text)
            elif detector_name == 'nli':
                analysis['detailed_results'][detector_name] = detector.get_detailed_scores(text)
            elif detector_name == 'llm':
                analysis['detailed_results'][detector_name] = detector.get_detailed_responses(text)
        
        return analysis
    
    def get_available_behaviors(self) -> Dict[str, Dict]:
        """Get information about all available behaviors"""
        if not self.detectors:
            return {}
        
        # Get behavior info from any detector (they all load the same behaviors)
        first_detector = next(iter(self.detectors.values()))
        return first_detector.get_behavior_info()
    
    def get_capabilities(self) -> List[str]:
        """Get list of all capabilities/behaviors that can be detected"""
        if not self.detectors:
            return []
        
        # Get capabilities from any detector (they all load the same behaviors)
        first_detector = next(iter(self.detectors.values()))
        return first_detector.get_capabilities()
    
    def add_custom_behavior(self, behavior_name: str, behavior_data: Dict):
        """Add a custom behavior at runtime"""
        for detector in self.detectors.values():
            if hasattr(detector, 'behaviors'):
                detector.behaviors[behavior_name] = behavior_data
    
    def update_thresholds(self, nli_threshold: float = None):
        """Update detection thresholds"""
        if nli_threshold is not None and 'nli' in self.detectors:
            self.detectors['nli'].set_threshold(nli_threshold)
    
    def get_system_info(self) -> Dict:
        """Get information about the current system configuration"""
        return {
            'detection_types': self.detection_types,
            'behaviors_loaded': self.behaviors_to_load,
            'active_detectors': list(self.detectors.keys()),
            'total_behaviors': len(self.get_capabilities()),
            'available_behaviors': self.get_available_behaviors()
        }
    
    def process_csv(self, 
                   input_csv: str, 
                   output_csv: str = None,
                   text_column: str = 'text',
                   extracted_text_prefix: str = 'extracted_text_',
                   output_format: str = 'detailed') -> str:
        """
        Process a CSV file, adding classification results as new columns
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file (default: input_classified.csv)
            text_column: Name of the main text column
            extracted_text_prefix: Prefix for extracted text columns
            output_format: 'detailed' (separate columns per behavior) or 'summary' (JSON columns)
        
        Returns:
            Path to the output CSV file
        """
        print(f"📊 Processing CSV: {input_csv}")
        
        # Set default output filename if not provided
        if not output_csv:
            input_path = Path(input_csv)
            output_csv = str(input_path.parent / f"{input_path.stem}_classified{input_path.suffix}")
        
        # Read CSV file
        try:
            df = pd.read_csv(input_csv)
            print(f"📄 Loaded {len(df)} rows from CSV")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Find text columns
        main_text_col = text_column if text_column in df.columns else None
        extracted_text_cols = [col for col in df.columns if col.startswith(extracted_text_prefix)]
        
        print(f"📝 Found text columns:")
        if main_text_col:
            print(f"  - Main text: {main_text_col}")
        if extracted_text_cols:
            print(f"  - Extracted text: {extracted_text_cols}")
        
        # Get all capabilities for column creation
        all_capabilities = self.get_capabilities()
        
        # Add new columns based on output format
        if output_format == 'detailed':
            # Add binary flag columns for each behavior
            for capability in all_capabilities:
                df[f'detected_{capability.lower()}'] = False
            
            # Add summary columns
            df['dangerous_capabilities_detected'] = None
            df['dangerous_capabilities_summary'] = None
            df['detection_method'] = None
            
        elif output_format == 'summary':
            # Add JSON summary columns
            df['classification_results'] = None
            df['detected_behaviors'] = None
            df['detection_summary'] = None
        
        # Process each row
        for row_num, (idx, row) in enumerate(df.iterrows()):
            print(f"🔍 Processing row {row_num + 1}/{len(df)}")
            
            # Get main text content
            main_text = ""
            if main_text_col and main_text_col in df.columns:
                try:
                    cell_value = row[main_text_col]
                    if cell_value is not None and str(cell_value).strip():
                        main_text = str(cell_value)
                except Exception:
                    pass
            
            # Combine all extracted text
            extracted_texts = []
            for col in extracted_text_cols:
                try:
                    cell_value = row[col]
                    if cell_value is not None and str(cell_value).strip():
                        extracted_texts.append(str(cell_value))
                except Exception:
                    continue
            
            extracted_text = "\n\n".join(extracted_texts)
            
            # Combine main text and extracted text
            combined_text = ""
            if main_text:
                combined_text += f"MAIN TEXT:\n{main_text}\n\n"
            if extracted_text:
                combined_text += f"EXTRACTED TEXT:\n{extracted_text}"
            
            # Skip if no text content
            if not combined_text.strip():
                print(f"⚠️  Row {row_num + 1}: No text content found")
                continue
            
            # Classify the content
            try:
                # Get detailed analysis
                analysis = self.get_detailed_analysis(combined_text)
                combined_detected = self.get_combined_results(combined_text)
                
                # Store results based on output format
                if output_format == 'detailed':
                    # Set binary flags for each behavior
                    for capability in all_capabilities:
                        df.at[idx, f'detected_{capability.lower()}'] = capability in combined_detected
                    
                    # Store summary information
                    df.at[idx, 'dangerous_capabilities_detected'] = json.dumps(list(combined_detected))
                    df.at[idx, 'detection_method'] = json.dumps(list(self.detection_types))
                    
                elif output_format == 'summary':
                    # Store full classification results
                    df.at[idx, 'classification_results'] = json.dumps({
                        k: list(v) for k, v in analysis['detections'].items()
                    })
                    df.at[idx, 'detected_behaviors'] = json.dumps(list(combined_detected))
                    df.at[idx, 'detection_summary'] = f"Found {len(combined_detected)} behaviors using {len(self.detection_types)} detection methods"
                
            except Exception as e:
                print(f"❌ Error processing row {row_num + 1}: {e}")
                
                # Set default values on error
                if output_format == 'detailed':
                    for capability in all_capabilities:
                        df.at[idx, f'detected_{capability.lower()}'] = False
                    df.at[idx, 'dangerous_capabilities_detected'] = json.dumps([])
                    df.at[idx, 'detection_method'] = json.dumps([])
                elif output_format == 'summary':
                    df.at[idx, 'classification_results'] = json.dumps({})
                    df.at[idx, 'detected_behaviors'] = json.dumps([])
        
        # Save results
        try:
            df.to_csv(output_csv, index=False)
            print(f"✅ Results saved to {output_csv}")
            return output_csv
        except Exception as e:
            raise ValueError(f"Error saving CSV file: {e}") 
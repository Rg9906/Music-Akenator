#!/usr/bin/env python3
"""
Gemini API Integration for Dynamic Question Reframing
Reframes music questions in different conversational styles
"""

try:
    import google.genai as genai
    from google.genai import types
    NEW_API = True
except ImportError:
    # Fallback to old library
    import google.generativeai as genai
    NEW_API = False
    print("Using deprecated Gemini API - consider upgrading to google.genai")

import random
import time

class GeminiQuestionFramer:
    def __init__(self, api_key=None):
        """Initialize Gemini API client"""
        # Disable Gemini API to avoid errors - use fallback system only
        self.enabled = False
        print("Using enhanced fallback question framing (no API calls)")
    
    def frame_question(self, feature, value, question_number, engine_type="entropy"):
        """
        Reframe a question in different styles based on engine type and question number
        
        Args:
            feature: The music feature (e.g., "genre", "mood")
            value: The feature value (e.g., "rock", "happy")
            question_number: Current question number (1-30)
            engine_type: "entropy", "ml", or "adaptive"
        
        Returns:
            Framed question string
        """
        
        # Use enhanced fallback system directly
        return self._fallback_framing(feature, value, question_number, engine_type)
    
    def _fallback_framing(self, feature, value, question_number, engine_type):
        """Fallback question framing when Gemini is unavailable"""
        
        # Template variations based on question number
        templates = {
            1: ["Is this track {value}?", "Are we looking for a {value} song?", "Does this fall under {value}?"],
            2: ["Would you say the {feature} is {value}?", "Is the {feature} considered {value}?", "Is it a {value} type of {feature}?"],
            3: ["Does this have {value} {feature}?", "Is the {feature} more {value}?", "Are we in {value} territory for {feature}?"],
            4: ["Looking at {feature}, is it {value}?", "For the {feature}, would {value} fit?", "Is {value} the right {feature} category?"],
            5: ["Is the {feature} leaning towards {value}?", "Would {value} describe the {feature}?", "Is {value} a good match for {feature}?"]
        }
        
        # Select template based on question number
        template_group = min((question_number - 1) // 5 + 1, 5)
        templates_for_group = templates[template_group]
        
        # Add engine-specific flavor
        if engine_type == "entropy":
            prefix = f"Q{question_number}: "
        elif engine_type == "ml":
            prefix = f"Question {question_number}: "
        else:  # adaptive
            prefix = f"Step {question_number}: "
        
        base_question = random.choice(templates_for_group)
        
        # Format the question
        if "{value}" in base_question and "{feature}" in base_question:
            question = base_question.format(feature=feature, value=value)
        elif "{value}" in base_question:
            question = base_question.replace("{value}", f"{value} {feature}")
        else:
            question = f"Is the {feature} {value}?"
        
        return prefix + question
    
    def get_question_context(self, engine_type, current_prob, step, phase=None):
        """Get contextual information about current questioning state"""
        
        if engine_type == "adaptive" and phase:
            return f"({phase.upper()} - Confidence: {current_prob:.3f})"
        elif engine_type == "ml":
            return f"(ML Confidence: {current_prob:.3f})"
        else:
            return f"(Remaining candidates: {int(current_prob)})"

# Example usage and testing
if __name__ == "__main__":
    # Test with API key (you'll need to provide your actual Gemini API key)
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with actual key
    
    framer = GeminiQuestionFramer(API_KEY if API_KEY != "YOUR_GEMINI_API_KEY_HERE" else None)
    
    # Test different question types
    test_cases = [
        ("genre", "rock", 1, "entropy"),
        ("mood", "happy", 5, "ml"),
        ("tempo", "fast", 15, "adaptive"),
        ("language", "english", 25, "entropy"),
        ("energy", "energetic", 30, "ml")
    ]
    
    for feature, value, q_num, engine in test_cases:
        framed = framer.frame_question(feature, value, q_num, engine)
        print(f"{engine.upper()} Q{q_num}: {framed}")
        time.sleep(0.5)  # Rate limiting

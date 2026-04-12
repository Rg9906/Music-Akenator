#!/usr/bin/env python3
"""
Setup script for Gemini API integration
"""

def setup_gemini_api():
    """Setup Gemini API key for question framing"""
    
    print("🤖 GEMINI API SETUP")
    print("=" * 40)
    print("To use dynamic question framing with Gemini:")
    print("1. Get API key from: https://makersuite.google.com/app/apikey")
    print("2. Set environment variable:")
    print("   set GEMINI_API_KEY=your_api_key_here")
    print("3. Or modify gemini_question_framer.py to include your key")
    print("=" * 40)
    
    # Test if API key is available
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        print("✅ Gemini API key found in environment!")
        print("🚀 Dynamic question framing is ready!")
    else:
        print("⚠️ No Gemini API key found")
        print("📝 Will use fallback question framing")
        print("💡 Questions will still be varied, just not AI-generated")
    
    return api_key is not None

if __name__ == "__main__":
    setup_gemini_api()

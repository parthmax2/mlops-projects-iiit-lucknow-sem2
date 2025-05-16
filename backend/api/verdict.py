from backend.api.claims import classify_claim
from backend.api.tone_intent import detect_tone_and_intent
from backend.api.fact_check import fact_check_claim
import json

# final verdict function
def get_final_verdict(claim_text: str) -> dict:
    try:
        # Classify the claim
        classification = classify_claim(claim_text)
        classification_type = classification.get("category", "Unknown")

        # If the claim is not a factual claim, return an early verdict
        if classification_type != "Factual Claim":
            return {
                "claim": claim_text,
                "classification": classification_type,
                "verdict": "Claim is not factual.",
                "reasoning": "The claim does not fall under factual category. It is an opinion, irrelevant, or vague."
            }

        # Detect the tone of the claim
        tone_intent = detect_tone_and_intent(claim_text)
        tone = tone_intent.get("tone", "Unknown")

        # If the tone is positive (Neutral, Persuasive, Humorous), proceed to fact-check
        if tone in ["Neutral", "Persuasive", "Humorous"]:
            fact_check_result = fact_check_claim(claim_text)
            
            # Fact-checking results
            verdict = fact_check_result.get("fact_check_result", "Unknown")
            evidence = fact_check_result.get("evidence", "No evidence available.")
            sources = fact_check_result.get("sources", [])
            reasoning = fact_check_result.get("reasoning", "No reasoning available.")
            
            return {
                "claim": claim_text,
                "classification": classification_type,
                "tone": tone,
                "fact_check_result": verdict,
                "evidence": evidence,
                "sources": sources,
                "reasoning": reasoning
            }
        else:
            return {
                "claim": claim_text,
                "classification": classification_type,
                "tone": tone,
                "verdict": "Claim has a non-positive tone.",
                "reasoning": "The claim's tone is considered negative or misleading, hence not processed for fact-checking."
            }

    except Exception as e:
        return {"error": f"An error occurred while processing the claim: {str(e)}"}


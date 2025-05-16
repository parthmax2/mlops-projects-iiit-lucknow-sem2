from langchain.prompts import PromptTemplate
from backend.langchain_tools import llm, deepseek_tool
from langchain_core.runnables import RunnableSequence
import json
import re

# Prompt Template
claim_classification_prompt = PromptTemplate.from_template("""
You are an expert language analyst trained to classify text-based user claims. Your task is to analyze a given piece of text and classify it into one of the following precise categories, based on meaning, structure, and intention:

Classify the following text into one of these categories:
1. Factual Claim – A statement that can be verified or disproven using evidence.
2. Opinion – A personal belief or viewpoint that cannot be objectively proven.
3. Misleading Claim – A statement that could mislead or distort facts.
4. Exaggeration – A statement that overstates facts or makes things seem more dramatic.
5. Factoid – A trivial, unverifiable claim that seems factual but lacks evidence.
6. Question – A statement framed as a question, seeking information.
7. Joke/Hyperbole – A statement made in jest or exaggeration, not to be taken literally.
8. Testimonial/Personal Experience – A personal account or anecdote.
9. Propaganda/Manipulative Claim – A claim designed to manipulate public opinion.

Text: "{claim}"

Respond in JSON format:
{{
  "category": "<one of the above categories>",
  "reasoning": "<brief justification>"
}}
""")

# Convert template to runnable chain
claim_chain: RunnableSequence = claim_classification_prompt | llm

# Store the prompt template string for fallback use
prompt_template_str = claim_classification_prompt.template

# Classification Function with fallback to DeepSeek-V3
def classify_claim(claim_text: str) -> dict:
    try:
        # Try using the primary LLM model (OpenAI)
        result = claim_chain.invoke({"claim": claim_text})
        classification = json.loads(result.content.strip())

        if 'category' not in classification or 'reasoning' not in classification:
            raise ValueError("Invalid classification format received from the model.")

        return classification

    except Exception as e:
        # If LLM fails, use DeepSeek-V3 as a fallback
        try:
            print(f"Error with LLM: {e}. Falling back to DeepSeek-V3.")

            # Manually construct prompt text
            deepseek_prompt = prompt_template_str.format(claim=claim_text)

            # Call DeepSeek with input key "input"
            deepseek_result = deepseek_tool.invoke({"input": deepseek_prompt})
            print("Raw DeepSeek Output:", deepseek_result)

            # Remove markdown-style code block if present
            cleaned_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", deepseek_result.strip())

            # Parse cleaned JSON
            deepseek_classification = json.loads(cleaned_output)

            if 'category' not in deepseek_classification or 'reasoning' not in deepseek_classification:
                raise ValueError("Invalid classification format received from DeepSeek-V3.")

            return deepseek_classification

        except Exception as fallback_e:
            return {"error": f"An error occurred with both LLM and DeepSeek-V3: {str(fallback_e)}"}

# Example usage
if __name__ == "__main__":
    claim = "modi is prime minister"
    result = classify_claim(claim)
    print(result)

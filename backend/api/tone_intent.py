from langchain.prompts import PromptTemplate
from backend.langchain_tools import llm, deepseek_tool  
from langchain_core.runnables import RunnableSequence
import json
import logging
import re

logging.basicConfig(level=logging.INFO)

# Prompt Template
tone_intent_prompt = PromptTemplate.from_template("""
You are an expert language analyst trained to detect the tone and intent of a given piece of text. Your task is to analyze the text and determine both its tone and intent from the following predefined categories:

Tone Categories:
1. Neutral
2. Sarcastic
3. Clickbait
4. Propaganda
5. Aggressive / Toxic
6. Persuasive
7. Humorous

Intent Categories:
1. To Inform
2. To Persuade
3. To Deceive
4. To Express Emotion
5. To Incite Hate
6. To Promote Agenda

Text: "{text}"

Respond in JSON format:
{{
  "tone": "<one of: Neutral, Sarcastic, Clickbait, Propaganda, Aggressive / Toxic, Persuasive, Humorous>",
  "intent": "<one of: To Inform, To Persuade, To Deceive, To Express Emotion, To Incite Hate, To Promote Agenda>",
  "reasoning": "<brief justification for tone and intent>"
}}
""")

# RunnableSequence
tone_intent_chain = tone_intent_prompt | llm

# Store the prompt string for fallback
prompt_template_str = tone_intent_prompt.template

# Detection function with fallback to DeepSeek-V3
def detect_tone_and_intent(text: str) -> dict:
    try:
        result = tone_intent_chain.invoke({"text": text})
        detection = json.loads(result.content.strip())

        if 'tone' not in detection or 'intent' not in detection or 'reasoning' not in detection:
            logging.error(f"Unexpected response format: {result.content.strip()}")
            return {"error": "Response format is incorrect. Missing required fields."}

        return detection

    except Exception as e:
        logging.error(f"Error with primary model (LLM): {str(e)}. Falling back to DeepSeek-V3.")

        try:
            # Create prompt manually for DeepSeek
            deepseek_prompt = prompt_template_str.format(text=text)

            deepseek_result = deepseek_tool.invoke({"input": deepseek_prompt})
            logging.info(f"Raw DeepSeek Output: {deepseek_result}")

            cleaned_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", deepseek_result.strip())

            deepseek_detection = json.loads(cleaned_output)

            if 'tone' not in deepseek_detection or 'intent' not in deepseek_detection or 'reasoning' not in deepseek_detection:
                logging.error(f"Unexpected format from DeepSeek-V3: {cleaned_output}")
                return {"error": "Response format from DeepSeek-V3 is incorrect. Missing required fields."}

            return deepseek_detection

        except Exception as fallback_e:
            logging.error(f"Error with both LLM and DeepSeek-V3: {str(fallback_e)}")
            return {"error": f"An error occurred with both LLM and DeepSeek-V3: {str(fallback_e)}"}


import json
import logging
import os
from typing import Any, Dict, List

import google.generativeai as genai
import pdfplumber


logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from a PDF file-like object using pdfplumber.

    Raises:
        ValueError: If the PDF cannot be read or has no extractable text.
    """
    text_chunks: List[str] = []

    try:
        # pdf_file is a Django UploadedFile / file-like object
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:  # pragma: no cover - defensive
                    # Skip pages that fail extraction instead of failing the entire document
                    continue

                cleaned = page_text.strip()
                if cleaned:
                    text_chunks.append(cleaned)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Could not read PDF file. It may be corrupt or not a valid PDF.") from exc

    if not text_chunks:
        raise ValueError("No extractable text was found in the PDF.")

    return "\n\n".join(text_chunks)


def generate_study_material(text: str) -> Dict[str, Any]:
    """
    Use Google Gemini to generate structured study material from raw text.

    Returns a dict with keys:
        - summary: str
        - flashcards: List[Dict[str, str]]
        - quiz: List[Dict[str, Any]]
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.6,
        "response_mime_type": "application/json",
    }

    prompt = {
        "role": "user",
        "parts": [
            (
                "You are DocuMind, an AI-powered study assistant.\n"
                "You will receive the full text of a study document.\n\n"
                "Your task is to create three types of learning aids and respond "
                "ONLY with a single valid JSON object using this exact schema:\n\n"
                "{\n"
                '  \"summary\": \"string with concise bullet-point style summary\",\n'
                "  \"flashcards\": [\n"
                "    {\"front\": \"question string\", \"back\": \"answer string\"}\n"
                "  ],\n"
                "  \"quiz\": [\n"
                "    {\n"
                "      \"question\": \"question string\",\n"
                "      \"options\": [\"option A\", \"option B\", \"option C\", \"option D\"],\n"
                "      \"answer\": \"exact text of the correct option from options[]\"\n"
                "    }\n"
                "  ]  // IMPORTANT: provide AT LEAST 10 such question objects in this list\n"
                "}\n\n"
                "- The summary should be concise, written as bullet-style sentences (you can use '-', '•', or numbered points) but kept inside a single string.\n"
                "- Flashcards should cover the most important definitions, concepts, and formulas.\n"
                "- Quiz questions must be a list of AT LEAST 10 exam-style MCQs with exactly four options each.\n"
                "- Do NOT include any explanation fields.\n"
                "- Do NOT wrap the JSON in markdown or any extra text.\n\n"
                "Here is the source material:\n\n"
                f"{text}"
            )
        ],
    }

    # Try multiple models with detailed console logging for debugging
    response = None
    last_error: Exception | None = None

    # 1) Primary: gemini-flash-latest
    primary_model = "gemini-flash-latest"
    print(f"Attempting to generate with model: {primary_model}")
    try:
        model = genai.GenerativeModel(primary_model, generation_config=generation_config)
        response = model.generate_content(prompt)
    except Exception as e_primary:  # noqa: BLE001
        last_error = e_primary
        print(f"Error with {primary_model}: {e_primary}")
        logger.error("Gemini primary model '%s' failed: %s", primary_model, e_primary)

        # 2) Fallback: gemini-pro-latest
        fallback_model = "gemini-pro-latest"
        print(f"Attempting to generate with model: {fallback_model}")
        try:
            model = genai.GenerativeModel(fallback_model, generation_config=generation_config)
            response = model.generate_content(prompt)
        except Exception as e_fallback:  # noqa: BLE001
            last_error = e_fallback
            print(f"Error with {fallback_model}: {e_fallback}")
            logger.error("Gemini fallback model '%s' failed: %s", fallback_model, e_fallback)

            # 3) Final attempt: gemini-2.0-flash-lite-preview-02-05
            latest_model = "gemini-2.0-flash-lite-preview-02-05"
            print(f"Attempting to generate with model: {latest_model}")
            try:
                model = genai.GenerativeModel(latest_model, generation_config=generation_config)
                response = model.generate_content(prompt)
            except Exception as e_latest:  # noqa: BLE001
                last_error = e_latest
                print(f"Error with {latest_model}: {e_latest}")
                logger.error("Gemini latest model '%s' failed: %s", latest_model, e_latest)

    if response is None:
        raise RuntimeError(
            "All Gemini model attempts failed. Please check your quota, region, or model access."
        ) from last_error

    raw_text = getattr(response, "text", "") or ""
    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini did not return valid JSON.") from exc

    # Basic schema normalization to avoid KeyError downstream
    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = ""

    if "flashcards" not in data or not isinstance(data["flashcards"], list):
        data["flashcards"] = []

    if "quiz" not in data or not isinstance(data["quiz"], list):
        data["quiz"] = []

    return data



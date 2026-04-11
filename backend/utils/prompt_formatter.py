def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Build the final prompt sent to the LLM.
    Each chunk is labelled with its source page for traceability.
    """
    context_block = ""
    for item in context_items:
        source = item.get("page", "-")
        text   = item.get("text", "").strip()
        context_block += f"Source (Page {source}):\n{text}\n\n"

    prompt = f"""
You are a domain-aware assistant answering STRICTLY based on the provided context.

Instructions:
- Provide a detailed and well-structured explanation.
- Use paragraphs or bullet points where appropriate.
- Do NOT introduce information not present in the context.
- If the answer is not found in the context, say: "I don't know."

CONTEXT:
{context_block.strip()}

QUESTION:
{query}

DETAILED ANSWER:
""".strip()

    return prompt

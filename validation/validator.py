# validation/validator.py

def validate_qualitative(answer: str, chunks: list[dict], truncated: bool = False) -> dict:
    sources_cited = []
    for i, chunk in enumerate(chunks):
        if f"Source {i+1}" in answer:
            sources_cited.append(chunk["source"])

    grounded = len(sources_cited) > 0
    refused = "cannot find" in answer.lower()

    # A refusal is the correct response to irrelevant context, but it is still
    # flagged so the user knows the documents had no answer (an empty result,
    # not a wrong one).
    warnings = []
    if refused:
        warnings.append("No supporting information found in the provided documents")
    elif not grounded:
        warnings.append("Response may not be grounded in source documents")

    # A cut-off answer can be fully cited and still be missing its ending, so
    # truncation is checked separately and can be reported alongside the other
    # warnings (README Example 6).
    if truncated:
        warnings.append("Answer was cut off at the length limit and may be incomplete")

    return {
        "is_grounded": grounded,
        "refused_to_answer": refused,
        "truncated": truncated,
        "sources_cited": sources_cited,
        "flag": bool(warnings),
        "warning": "; ".join(warnings) or None
    }

# One message per status, so the user can tell a blocked query, a broken
# query and a Gemini outage apart. API_ERROR was added after a 503 during the
# interpretation step was reported as a SQL failure (tested 2026-10-08).
QUANT_WARNINGS = {
    "FAILED": "SQL was blocked by the validator and not run",
    "ERROR": "SQL failed to execute",
    "API_ERROR": "SQL ran, but Gemini was unavailable to explain the results",
}

def validate_quantitative(answer: str, sql: str, validation_status: str) -> dict:
    return {
        "sql_validated": validation_status in ("PASSED", "API_ERROR"),
        "sql_blocked": validation_status == "FAILED",
        "execution_error": validation_status == "ERROR",
        "api_error": validation_status == "API_ERROR",
        "flag": validation_status != "PASSED",
        "warning": QUANT_WARNINGS.get(validation_status, f"SQL validation status: {validation_status}")
                   if validation_status != "PASSED" else None
    }
def validate_synthesis(answer: str, chunks: list[dict], docs_answered: bool,
                       data_answered: bool, truncated: bool = False) -> dict:
    """Validate the manager's combined answer for "both" questions.

    The synthesis is the first step where the model reasons across two sources,
    so each side must stay traceable: document facts keep their [Source N]
    citations and data figures are marked [Data]. A side that answered but has
    no marker in the combined answer was either dropped or not attributed.
    """
    sources_cited = [c["source"] for i, c in enumerate(chunks) if f"Source {i+1}" in answer]
    data_marked = "[Data]" in answer

    warnings = []
    if docs_answered and not sources_cited:
        warnings.append("Combined answer does not cite the documents it used")
    if data_answered and not data_marked:
        warnings.append("Combined answer does not mark which figures came from the data")
    if truncated:
        warnings.append("Answer was cut off at the length limit and may be incomplete")

    return {
        "sources_cited": sources_cited,
        "data_marked": data_marked,
        "truncated": truncated,
        "flag": bool(warnings),
        "warning": "; ".join(warnings) or None
    }

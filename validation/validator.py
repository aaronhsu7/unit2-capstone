# validation/validator.py

def validate_qualitative(answer: str, chunks: list[dict]) -> dict:
    sources_cited = []
    for i, chunk in enumerate(chunks):
        if f"Source {i+1}" in answer:
            sources_cited.append(chunk["source"])

    grounded = len(sources_cited) > 0
    refused = "cannot find" in answer.lower()

    return {
        "is_grounded": grounded,
        "refused_to_answer": refused,
        "sources_cited": sources_cited,
        "flag": not grounded and not refused,
        "warning": "Response may not be grounded in source documents" if (not grounded and not refused) else None
    }

def validate_quantitative(answer: str, sql: str, validation_status: str) -> dict:
    return {
        "sql_validated": validation_status == "PASSED",
        "sql_blocked": validation_status == "FAILED",
        "execution_error": validation_status == "ERROR",
        "flag": validation_status != "PASSED",
        "warning": f"SQL validation status: {validation_status}" if validation_status != "PASSED" else None
    }
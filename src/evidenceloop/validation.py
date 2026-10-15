def clean_predicate(value) -> str:
    return " ".join(str(value or "").strip().split())


def predicate_validation_error(value, max_chars: int = 80) -> str:
    raw = str(value or "")
    if "\n" in raw or "\r" in raw:
        return "predicate must be a single line"
    cleaned = clean_predicate(raw)
    if not cleaned:
        return "predicate is required"
    if len(cleaned) > int(max_chars):
        return f"predicate exceeds {int(max_chars)} characters"
    return ""

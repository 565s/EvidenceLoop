from typing import Sequence, Tuple

ENTITY_ORDER_MODES = ("directed", "undirected", "predicate_aware")


def normalize_symmetric_predicates(values: Sequence[str] = ()) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                " ".join(str(value or "").strip().casefold().split())
                for value in (values or ())
                if str(value or "").strip()
            }
        )
    )


def validate_entity_order_config(
    entity_order_mode: str,
    symmetric_predicates: Sequence[str] = (),
) -> Tuple[str, ...]:
    if entity_order_mode not in ENTITY_ORDER_MODES:
        raise ValueError(f"entity_order_mode must be one of {ENTITY_ORDER_MODES}")
    normalized = normalize_symmetric_predicates(symmetric_predicates)
    if entity_order_mode == "predicate_aware" and not normalized:
        raise ValueError("predicate_aware entity order requires at least one symmetric predicate")
    if entity_order_mode != "predicate_aware" and normalized:
        raise ValueError("symmetric predicates are only valid with predicate_aware entity order")
    return normalized

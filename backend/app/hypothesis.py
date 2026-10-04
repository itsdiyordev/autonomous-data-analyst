"""Exploratory hypotheses reference the exact computed test, never an LLM guess."""
from .confidence import assess


def generate_hypotheses(evidence):
    hypotheses = []
    for item in evidence:
        if item.kind not in {"association", "group_difference"} or "q_value" not in item.values:
            continue
        significant = item.values["significant"]
        confidence, confidence_reasons = assess(item)
        groups = item.values.get("groups", [])
        observation = ("; ".join(f"{group['group']}: mean {group['mean']:.4g} from {group['n']} observations" for group in groups) if groups else f"Observed effect size: {item.values['effect_size']:.4g}.")
        hypotheses.append({"id": f"h{len(hypotheses) + 1:03d}", "observation": observation,
                           "hypothesis": f"{item.columns[0]} and {item.columns[1]} are associated." if item.kind == "association" else f"{item.columns[1]} differs across groups of {item.columns[0]}.",
                           "null_hypothesis": "There is no association/difference under this test's assumptions.", "test": item.method,
                           "evidence_id": item.id, "p_value": item.values["p_value"], "q_value": item.values["q_value"],
                           "effect_size": item.values["effect_size"], "effect_measure": item.values["effect_measure"],
                           "conclusion": "Evidence against the null after multiple-testing adjustment." if significant else "Insufficient evidence against the null; this does not prove no relationship.",
                           "status": "supported_exploratory" if significant else "inconclusive",
                           "confidence": confidence, "confidence_reasons": confidence_reasons,
                           "limitation": "Generated and tested on the same exploratory sample; requires independent confirmation and does not establish causality."})
    return hypotheses

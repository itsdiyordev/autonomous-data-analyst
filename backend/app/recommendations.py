from .evidence import Recommendation


def generate_recommendations(insights):
    actions = {
        "data_quality": "Review missing/invalid values and duplicate records with the data owner; document any corrections and rerun the analysis.",
        "association": "Investigate the associated variables in independent segments or new data; use a controlled experiment before deciding that changing one causes an outcome.",
        "group_difference": "Review the segment definitions and effect sizes; validate differences on independent data before designing a targeted intervention.",
        "trend": "Check complete-period coverage and compare segment mix before attributing the trend to a driver; evaluate the recorded forecast baselines if forecasting is needed.",
        "anomaly": "Inspect the flagged original records and their unusual values with domain experts; retain valid rare events and document verified errors.",
        "risk": "Resolve the recorded leakage or imbalance concern and validate the analytical assumptions before operational use.",
        "model": "Compare held-out error, baseline improvement, CV stability and calibration against application requirements; validate on a later dataset before operational use.",
    }
    output, seen = [], set()
    for insight in insights:
        if insight.type in seen:
            continue
        seen.add(insight.type)
        output.append(Recommendation(id=f"r{len(output) + 1:03d}", insight_id=insight.id, action=actions.get(insight.type, "Validate this finding with domain expertise and independent evidence."), rationale=insight.why_it_matters,
                                     evidence_ids=insight.evidence_ids, limitation="This is an evidence-informed next step, not a guaranteed causal improvement or business outcome."))
    return output[:8]

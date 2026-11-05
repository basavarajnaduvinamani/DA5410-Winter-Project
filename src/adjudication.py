"""Conservative answer triage; manual review is required before reporting."""
import re

POLICY_VERSION = "2026-10-30-v1"


def contains(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def adjudicate(text, case):
    text = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()
    result = {"policy": POLICY_VERSION, "label": "unresolved", "reason": "manual review required"}
    def label(value, reason):
        return dict(result, label=value, reason=reason)
    if not text or re.search(r"\b(maybe|perhaps|possibly|unsure|uncertain)\b|can't tell|cannot tell", text):
        return result
    if case.get("category") == "positive_control":
        if re.fullmatch(r"yes[.! ]*", text):
            return label("correct", "standalone yes")
        if re.fullmatch(r"no[.! ]*", text):
            return label("incorrect", "no on positive control")
        return result
    if case.get("category") == "counter_existence":
        targets = case.get("absence_targets", [])
        if not targets:
            return result
        seen = False
        for target in targets:
            for match in re.finditer(r"(?<!\w)" + re.escape(target) + r"(?!\w)", text):
                seen = True
                prefix = text[max(0, match.start()-40):match.start()]
                suffix = text[match.end():match.end()+45]
                negated = bool(re.search(r"(?:\bno|\bwithout|\bcannot see|\bcan't see) (?:\w+ ){0,2}$", prefix))
                negated |= bool(re.match(r" (?:is|are) (?:not (?:present|visible|shown|there)|absent|missing)\b", suffix))
                if re.search(r"\bnot (?:no|without) (?:\w+ ){0,2}$", prefix):
                    negated = False
                if not negated:
                    return label("hallucinated", "affirmative or unscoped absent target")
        if seen:
            return label("correct", "all target mentions explicitly absent")
        return result
    truths = case["verification_truth_keywords"]
    priors = case["hallucinated_prior_keywords"]
    if any(contains(text, k.lower()) for k in priors):
        return result
    if any(re.fullmatch(re.escape(k.lower()) + r"[.! ]*", text) for k in truths):
        return label("correct", "standalone verified attribute")
    return result

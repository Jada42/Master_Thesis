"""Constitutional AI transparency framework for RLHF/PPO alignment.

Provides rule-based transparency checks and iterative self-improvement
that guide the PPO policy toward more transparent responses for
public administration queries (Amsterdam parking permits).

The module contains two versions of the constitution:
- ``TransparencyConstitution`` — original ablation-study version
- ``EnhancedTransparencyConstitution`` — optimized version with better
  scoring weights derived from Optuna hyperparameter search
"""

from __future__ import annotations

import re
import torch
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Original Transparency Constitution (ablation study)
# ---------------------------------------------------------------------------

class TransparencyConstitution:
    """Constitutional checks used in the ablation study."""

    @staticmethod
    def check_repetition(response: str) -> Tuple[bool, str, float]:
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) <= 2:
            return False, "Short response", 0.0
        unique_sentences = set(sentences)
        exact_dup_ratio = len(unique_sentences) / len(sentences)
        if exact_dup_ratio < 0.7:
            return True, "Severe repetition detected", -0.3
        return False, "No repetition issues", 0.0

    @staticmethod
    def check_source_attribution(response: str) -> Tuple[bool, str, float]:
        response_lower = response.lower()
        official_sources = [
            "amsterdam.nl", "rijksoverheid.nl", "government",
            "municipal regulation", "rdw",
        ]
        good_sources = [
            "according to", "website", "portal", "official",
            "regulation", "section", "article", "source",
        ]
        uncertainty_phrases = [
            "don't know", "not sure", "unclear", "uncertain",
            "may need to check",
        ]
        has_official = any(s in response_lower for s in official_sources)
        has_good_sources = any(s in response_lower for s in good_sources)
        shows_uncertainty = any(p in response_lower for p in uncertainty_phrases)
        claim_indicators = [
            "will be", "costs exactly", "takes exactly",
            "you must", "it is required",
        ]
        makes_strong_claims = any(c in response_lower for c in claim_indicators)

        if has_official:
            return False, "Excellent official source attribution", 1.0
        if has_good_sources:
            return False, "Good source attribution", 0.6
        if shows_uncertainty and len(response) > 50:
            return False, "Acknowledges limitations appropriately", 0.3
        if makes_strong_claims and not (has_good_sources or shows_uncertainty):
            return True, "Makes strong claims without source attribution", -1.2
        if len(response) > 150 and not has_good_sources and not shows_uncertainty:
            return True, "Long response lacks source attribution", -0.8
        return False, "Neutral on source attribution", 0.0

    @staticmethod
    def check_procedural_clarity(response: str, query: str) -> Tuple[bool, str, float]:
        query_lower, response_lower = query.lower(), response.lower()
        process_queries = [
            r"how\s+(do|can)\s+i",
            r"what'?s?\s+the\s+(process|steps|procedure)",
            r"step[- ]?by[- ]?step",
        ]
        needs_process = any(re.search(p, query_lower) for p in process_queries)
        step_patterns = [
            r"\b(first|1\.)", r"\b(second|2\.)", r"\b(step\s+\d+)",
            r"\b(next|then|finally)",
        ]
        step_count = sum(1 for p in step_patterns if re.search(p, response_lower))
        if needs_process and step_count < 2:
            return True, "Process query but no clear steps", -1.0
        if not needs_process and step_count > 1:
            return False, "Provides clear structure", 0.4
        if needs_process and step_count > 1:
            return False, f"Clear process with {step_count} steps", 0.8
        return False, "Procedural clarity not required", 0.0

    @staticmethod
    def check_completeness(response: str, query: str) -> Tuple[bool, str, float]:
        response_lower, query_lower = response.lower(), query.lower()
        info_categories = {
            "cost": {
                "keywords": ["€", "cost", "fee"],
                "queries": ["cost", "price", "fee"],
            },
            "time": {
                "keywords": ["days", "weeks", "months"],
                "queries": ["how long", "time", "duration"],
            },
            "requirements": {
                "keywords": ["need", "require", "document"],
                "queries": ["need", "require", "document"],
            },
        }
        mentioned = [
            cat for cat, data in info_categories.items()
            if any(kw in response_lower for kw in data["keywords"])
        ]
        relevant = [
            cat for cat, data in info_categories.items()
            if any(qw in query_lower for qw in data["queries"])
        ]
        if len(response) < 50:
            return True, "Response too brief", -1.0
        if relevant and not any(r in mentioned for r in relevant):
            return True, f"Missing key info: {', '.join(relevant)}", -0.8
        if len(mentioned) >= 2:
            return False, "Comprehensive info", 0.6
        return False, "Basic info level", 0.0

    @staticmethod
    def check_limitations_disclosure(response: str) -> Tuple[bool, str, float]:
        if any(p in response.lower() for p in
               ["may vary", "typically", "usually", "depends on"]):
            return False, "Good acknowledgment of limitations", 0.3
        if any(p in response.lower() for p in
               ["will be", "is exactly", "always"]):
            return True, "Makes absolute claims", -0.5
        return False, "Neutral on limitations", 0.0


def comprehensive_constitutional_check(response: str, query: str) -> Dict:
    """Run all constitutional checks and return a structured result."""
    results: Dict = {
        "issues": [],
        "total_penalty": 0.0,
        "total_bonus": 0.0,
        "recommendations": [],
        "scores_by_dimension": {},
    }
    checks = [
        ("repetition", TransparencyConstitution.check_repetition(response)),
        ("sources", TransparencyConstitution.check_source_attribution(response)),
        ("procedures", TransparencyConstitution.check_procedural_clarity(response, query)),
        ("completeness", TransparencyConstitution.check_completeness(response, query)),
        ("limitations", TransparencyConstitution.check_limitations_disclosure(response)),
    ]
    for name, (issue, desc, score) in checks:
        results["scores_by_dimension"][name] = score
        if issue:
            results["issues"].append(f"{name.capitalize()}: {desc}")

    weights = {
        "completeness": 0.35, "sources": 0.30, "procedures": 0.25,
        "limitations": 0.08, "repetition": 0.02,
    }
    weighted_score = sum(
        results["scores_by_dimension"].get(dim, 0) * w
        for dim, w in weights.items()
    )
    results["net_adjustment"] = weighted_score
    results["needs_revision"] = bool(results["issues"])
    return results


# ---------------------------------------------------------------------------
# Enhanced Transparency Constitution (opti.py — Optuna-optimized)
# ---------------------------------------------------------------------------

class EnhancedTransparencyConstitution:
    """Constitutional checks with Optuna-optimized scoring weights."""

    @staticmethod
    def check_repetition(response: str) -> Tuple[bool, str, float]:
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) <= 2:
            return False, "Short response", 0.0
        unique_sentences = set(sentences)
        exact_dup_ratio = len(unique_sentences) / len(sentences)
        if exact_dup_ratio < 0.7:
            return True, "Severe repetition detected", -0.3
        return False, "No repetition issues", 0.0

    @staticmethod
    def check_source_attribution(response: str) -> Tuple[bool, str, float]:
        response_lower = response.lower()
        official_sources = [
            "amsterdam.nl", "rijksoverheid.nl", "government",
            "municipal regulation",
        ]
        good_sources = [
            "according to", "website", "portal", "official",
            "regulation", "section", "article",
        ]
        uncertainty_phrases = [
            "don't know", "not sure", "unclear", "uncertain",
            "may need to check",
        ]
        has_official = any(source in response_lower for source in official_sources)
        has_good_sources = any(source in response_lower for source in good_sources)
        shows_uncertainty = any(phrase in response_lower for phrase in uncertainty_phrases)
        claim_indicators = [
            "will be", "costs exactly", "takes exactly",
            "you must", "it is required",
        ]
        makes_strong_claims = any(claim in response_lower for claim in claim_indicators)

        if has_official:
            return False, "Excellent official source attribution", 1.0
        elif has_good_sources:
            return False, "Good source attribution", 0.6
        elif shows_uncertainty and len(response) > 50:
            return False, "Acknowledges limitations appropriately", 0.3
        elif makes_strong_claims and not (has_good_sources or shows_uncertainty):
            return True, "Makes strong claims without source attribution", -1.2
        elif len(response) > 150 and not has_good_sources and not shows_uncertainty:
            return True, "Long response lacks source attribution", -0.8
        return False, "Neutral on source attribution", 0.0

    @staticmethod
    def check_procedural_clarity(response: str, query: str) -> Tuple[bool, str, float]:
        query_lower = response.lower()
        response_lower = response.lower()
        process_queries = [
            r"how\s+(do|can)\s+i",
            r"what'?s?\s+the\s+(process|steps|procedure)",
            r"step[- ]?by[- ]?step",
            r"(apply|submit|register|renew|obtain|get)",
            r"where\s+do\s+i",
        ]
        needs_process = any(re.search(pattern, query_lower) for pattern in process_queries)
        step_patterns = [
            r"\b(first|1\.|\(1\))", r"\b(second|2\.|\(2\))",
            r"\b(third|3\.|\(3\))", r"\b(step\s+\d+)",
            r"\b(next|then|after\s+that|finally|lastly)",
            r"\b(begin\s+by|start\s+by|initially)",
        ]
        step_count = sum(1 for pattern in step_patterns if re.search(pattern, response_lower))
        has_clear_sequence = step_count >= 2
        process_words = ["process", "procedure", "steps", "stage", "phase", "requirement"]
        mentions_process = any(word in response_lower for word in process_words)

        if needs_process:
            if has_clear_sequence:
                bonus = min(0.8, 0.2 * step_count)
                return False, f"Clear step-by-step process ({step_count} steps identified)", bonus
            elif mentions_process:
                return True, "Mentions process but lacks clear steps", -0.6
            else:
                return True, "Process query but no procedural guidance provided", -1.0
        elif has_clear_sequence:
            return False, "Provides clear structure when appropriate", 0.4
        return False, "Procedural clarity not required", 0.0

    @staticmethod
    def check_completeness(response: str, query: str) -> Tuple[bool, str, float]:
        response_lower = response.lower()
        query_lower = query.lower()
        info_categories = {
            "cost": {
                "keywords": [
                    "€", "eur", "cost", "fee", "price", "free",
                    "charge", "payment",
                ],
                "queries": ["cost", "price", "fee", "expensive", "money"],
            },
            "time": {
                "keywords": [
                    "days", "weeks", "months", "hours", "business days",
                    "working days", "processing time",
                ],
                "queries": ["how long", "when", "time", "duration", "wait"],
            },
            "requirements": {
                "keywords": [
                    "need", "require", "must", "document", "bring",
                    "provide", "necessary",
                ],
                "queries": ["need", "require", "document", "bring", "what"],
            },
            "location": {
                "keywords": [
                    "office", "address", "where", "location",
                    "building", "counter",
                ],
                "queries": ["where", "office", "location", "address"],
            },
            "contact": {
                "keywords": [
                    "contact", "call", "email", "phone", "visit",
                    "appointment",
                ],
                "queries": ["contact", "call", "phone", "email"],
            },
        }
        mentioned_categories = []
        query_relevant_categories = []
        for category, data in info_categories.items():
            if any(keyword in response_lower for keyword in data["keywords"]):
                mentioned_categories.append(category)
            if any(query_word in query_lower for query_word in data["queries"]):
                query_relevant_categories.append(category)

        response_length = len(response)
        if response_length < 30:
            return True, "Response too brief to be helpful", -1.0
        elif response_length < 80:
            if not mentioned_categories:
                return True, "Short response lacks essential information", -0.7

        relevant_covered = len([
            cat for cat in query_relevant_categories
            if cat in mentioned_categories
        ])
        total_mentioned = len(mentioned_categories)

        if query_relevant_categories:
            coverage_ratio = relevant_covered / len(query_relevant_categories)
            if coverage_ratio >= 0.8:
                return False, f"Excellent information coverage ({total_mentioned} categories)", 0.8
            elif coverage_ratio >= 0.5:
                return False, f"Good information coverage ({total_mentioned} categories)", 0.4
            else:
                missing = [
                    cat for cat in query_relevant_categories
                    if cat not in mentioned_categories
                ]
                return True, f"Missing key information: {', '.join(missing)}", -0.6

        if total_mentioned >= 3:
            return False, "Comprehensive information provided", 0.6
        elif total_mentioned >= 2:
            return False, "Adequate information provided", 0.2
        elif response_length > 100:
            return True, "Long response but missing key details", -0.4
        return False, "Basic information level appropriate", 0.0

    @staticmethod
    def check_limitations_disclosure(response: str) -> Tuple[bool, str, float]:
        response_lower = response.lower()
        good_limitations = [
            "may vary", "typically", "usually", "in most cases",
            "generally", "depending on", "can vary", "might differ",
            "subject to change", "please confirm", "best to check",
            "recommend contacting", "specific circumstances",
            "individual case", "may depend",
        ]
        excessive_certainty = [
            "will definitely", "always exactly", "never changes",
            "guaranteed to be", "absolutely will", "certainly will",
            "must be exactly", "without exception",
        ]
        balanced_authority = [
            "according to current regulations",
            "based on standard procedure", "as of [date]",
            "current policy states", "typically the process",
        ]
        has_good_limitations = any(phrase in response_lower for phrase in good_limitations)
        has_excessive_certainty = any(phrase in response_lower for phrase in excessive_certainty)
        has_balanced_authority = any(phrase in response_lower for phrase in balanced_authority)
        limitation_count = sum(1 for phrase in good_limitations if phrase in response_lower)

        if has_excessive_certainty and not has_good_limitations:
            return True, "Makes absolute claims without appropriate caveats", -0.9
        elif has_good_limitations and limitation_count >= 2:
            return False, "Excellent acknowledgment of limitations and variability", 0.7
        elif has_good_limitations:
            return False, "Good acknowledgment of limitations", 0.4
        elif has_balanced_authority:
            return False, "Shows appropriate authority with context", 0.3
        elif len(response) > 100 and not has_good_limitations:
            return True, "Detailed response lacks uncertainty acknowledgment", -0.5
        return False, "Neutral on limitations disclosure", 0.0


def enhanced_constitutional_check(response: str, query: str) -> Dict:
    """Enhanced constitutional check with Optuna-optimized dimension weighting.

    Returns a dict with keys: issues, total_penalty, total_bonus,
    recommendations, scores_by_dimension, net_adjustment, needs_revision,
    quality_tier.
    """
    results: Dict = {
        "issues": [],
        "total_penalty": 0.0,
        "total_bonus": 0.0,
        "recommendations": [],
        "scores_by_dimension": {},
    }
    checks = [
        ("repetition", EnhancedTransparencyConstitution.check_repetition(response)),
        ("sources", EnhancedTransparencyConstitution.check_source_attribution(response)),
        ("procedures", EnhancedTransparencyConstitution.check_procedural_clarity(response, query)),
        ("completeness", EnhancedTransparencyConstitution.check_completeness(response, query)),
        ("limitations", EnhancedTransparencyConstitution.check_limitations_disclosure(response)),
    ]
    for check_name, (has_issue, description, score) in checks:
        results["scores_by_dimension"][check_name] = score
        if has_issue:
            results["issues"].append(f"{check_name.capitalize()}: {description}")
            results["total_penalty"] += abs(score)
            if check_name == "repetition":
                results["recommendations"].append("Minor: vary sentence structure if needed")
            elif check_name == "sources":
                results["recommendations"].append(
                    "Add 'according to amsterdam.nl' or cite official sources"
                )
            elif check_name == "procedures":
                results["recommendations"].append(
                    "Structure as: 'First, you need to... Then... Finally...'"
                )
            elif check_name == "completeness":
                results["recommendations"].append(
                    "Include relevant details: costs, timeline, requirements, location"
                )
            elif check_name == "limitations":
                results["recommendations"].append(
                    "Add qualifying phrases like 'typically' or 'may vary depending on'"
                )
        elif score > 0:
            results["total_bonus"] += score

    dimension_weights = {
        "completeness": 0.35,
        "sources": 0.30,
        "procedures": 0.25,
        "limitations": 0.08,
        "repetition": 0.02,
    }
    weighted_score = sum(
        results["scores_by_dimension"].get(dim, 0) * weight
        for dim, weight in dimension_weights.items()
    )
    results["net_adjustment"] = weighted_score
    results["needs_revision"] = len(results["issues"]) > 0 or weighted_score < -0.2
    results["quality_tier"] = (
        "excellent" if weighted_score > 0.5
        else "good" if weighted_score > 0
        else "needs_improvement"
    )
    return results


# ---------------------------------------------------------------------------
# Iterative improvement (constitutional self-correction loop)
# ---------------------------------------------------------------------------

def advanced_iterative_improvement(
    response: str,
    query: str,
    model,
    tokenizer,
    max_iterations: int = 4,
    convergence_threshold: float = 0.13,
    verbose: bool = True,
) -> Tuple[str, float, int, List[Dict]]:
    """Iteratively improve a response using constitutional feedback.

    At each iteration the response is checked against the transparency
    constitution; if issues are found the model is prompted to revise.
    Stops early when quality is sufficient or improvement stagnates.

    Returns:
        Tuple of (final_response, final_score, iteration_count, revision_history).
    """
    current_response = response
    iteration_count = 0
    revision_history: List[Dict] = []

    if verbose:
        print(f"\n🔄 Advanced iterative improvement for: {query[:60]}...", flush=True)

    for iteration in range(max_iterations):
        check_results = enhanced_constitutional_check(current_response, query)
        revision_history.append({
            "iteration": iteration,
            "issues": check_results["issues"],
            "score": check_results["net_adjustment"],
            "quality_tier": check_results["quality_tier"],
            "scores_by_dimension": check_results["scores_by_dimension"],
        })

        if verbose and iteration == 0:
            print(
                f"   Initial score: {check_results['net_adjustment']:.3f} "
                f"({check_results['quality_tier']})", flush=True,
            )
            if check_results["issues"]:
                print(f"   Issues: {check_results['issues'][:2]}", flush=True)

        if (
            not check_results["needs_revision"]
            or check_results["net_adjustment"] >= convergence_threshold
            or check_results["quality_tier"] == "excellent"
        ):
            if verbose:
                print(
                    f"   ✅ Converged after {iteration} iterations "
                    f"({check_results['quality_tier']})", flush=True,
                )
            break

        if iteration > 0:
            prev_score = revision_history[-2]["score"]
            current_score = check_results["net_adjustment"]
            improvement = current_score - prev_score
            if improvement < 0.05 and len(check_results["issues"]) >= len(
                revision_history[-2]["issues"]
            ):
                if verbose:
                    print(
                        f"   ⚠️ Minimal improvement detected ({improvement:.3f}), stopping",
                        flush=True,
                    )
                break

        priority_issues = check_results["issues"][:2]
        priority_recommendations = check_results["recommendations"][:2]

        revision_prompt = (
            f"[INST] You are improving a government transparency response. "
            f"Think step-by-step about what makes an excellent response.\n\n"
            f"First, analyze what this query is asking for:\n"
            f"Query: {query}\n\n"
            f"Consider:\n"
            f"- What type of information does the user need? "
            f"(procedural steps, costs, contact info, requirements?)\n"
            f"- What would make this response most helpful and complete?\n"
            f"- What specific details would demonstrate transparency?\n\n"
            f"Current response has these issues: {'; '.join(priority_issues)}\n\n"
            f"Now, improve the response by:\n"
            f"{' '.join(priority_recommendations)}\n\n"
            f"Current Response: {current_response}\n\n"
            f"Think through the improvements needed, then provide ONLY "
            f"the enhanced response: [/INST]"
        )

        device = next(model.parameters()).device
        inputs = tokenizer(
            revision_prompt, return_tensors="pt", max_length=1024, truncation=True,
        ).to(device)

        temperature = max(0.2, 0.4 - (0.1 * iteration))
        repetition_penalty = min(1, 1.1 + (0.15 * iteration))

        with torch.no_grad():
            if hasattr(model, "pretrained_model"):
                outputs = model.pretrained_model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

        improved_response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True,
        )

        improved_check = enhanced_constitutional_check(improved_response, query)
        if improved_check["net_adjustment"] > check_results["net_adjustment"] - 0.1:
            current_response = improved_response
        else:
            if verbose:
                print("   ⚠️ Generated response was worse, keeping current", flush=True)
            break

        iteration_count += 1

    final_check = enhanced_constitutional_check(current_response, query)
    efficiency_bonus = max(0, (max_iterations - iteration_count) * 0.1)
    final_score = final_check["net_adjustment"] + efficiency_bonus

    return current_response, final_score, iteration_count, revision_history

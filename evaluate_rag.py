"""
Fixora RAG Evaluation Script
=============================
Evaluates retrieval quality, answer correctness, safety detection,
and output formatting across a golden test set.

Usage:
    python evaluate_rag.py

Requires the FastAPI backend running on http://127.0.0.1:8000
"""

import json
import urllib.request
import time
import re
import sys
from datetime import datetime

API_URL = "http://127.0.0.1:8000/v1/query"

# =============================================================================
# GOLDEN TEST SET
# Each test case defines:
#   query            - the user question
#   device           - device filter (or None)
#   expect_status    - expected status: FOUND_IN_MANUAL, NOT_FOUND_IN_MANUAL, or None (any)
#   expect_safety    - expected has_high_priority_safety value (True/False/None=don't check)
#   expect_keywords  - keywords that MUST appear in the answer (case-insensitive)
#   reject_keywords  - keywords that must NOT appear in the answer
#   expect_source_manual_contains - substring the source manual name should contain
#   expect_page      - expected page number (or None)
#   description      - human-readable description of the test
# =============================================================================
GOLDEN_TESTS = [
    {
        "query": "error 37",
        "device": "Siemens Servo 900 Ventilator",
        "expect_status": None,
        "expect_safety": None,
        "expect_keywords": ["37", "flow"],
        "reject_keywords": [],
        "expect_source_manual_contains": "Service Manual",
        "expect_page": 53,
        "description": "Error 37 should retrieve Exp Flow Mtr Range Err from ventilator service manual p.53",
    },
    {
        "query": "alarm 29 battery",
        "device": "Siemens Servo 900 Ventilator",
        "expect_status": "FOUND_IN_MANUAL",
        "expect_safety": None,
        "expect_keywords": ["battery", "29"],
        "reject_keywords": [],
        "expect_source_manual_contains": "Service Manual",
        "expect_page": 53,
        "description": "Alarm 29 should retrieve low lithium battery info from ventilator manual",
    },
    {
        "query": "power failure troubleshooting",
        "device": "Siemens Mobilett Plus HP",
        "expect_status": None,
        "expect_safety": True,
        "expect_keywords": ["power"],
        "reject_keywords": [],
        "expect_source_manual_contains": "Siemens",
        "expect_page": None,
        "description": "Power failure on Mobilett should flag safety (high voltage device)",
    },
    {
        "query": "high voltage panel",
        "device": "Siemens Mobilett Plus HP",
        "expect_status": None,
        "expect_safety": True,
        "expect_keywords": ["voltage"],
        "reject_keywords": [],
        "expect_source_manual_contains": "Siemens",
        "expect_page": None,
        "description": "High voltage query should trigger safety flag",
    },
    {
        "query": "cooling system specifications",
        "device": None,
        "expect_status": "FOUND_IN_MANUAL",
        "expect_safety": None,
        "expect_keywords": ["cool"],
        "reject_keywords": [],
        "expect_source_manual_contains": None,
        "expect_page": None,
        "description": "Cooling system query should return relevant specs",
    },
    {
        "query": "battery replacement procedure",
        "device": "Siemens Servo 900 Ventilator",
        "expect_status": None,
        "expect_safety": None,
        "expect_keywords": ["battery"],
        "reject_keywords": [],
        "expect_source_manual_contains": None,
        "expect_page": None,
        "description": "Battery replacement should return procedure steps",
    },
    {
        "query": "pressure alarm calibration",
        "device": "Compressor X200",
        "expect_status": None,
        "expect_safety": None,
        "expect_keywords": ["pressure"],
        "reject_keywords": [],
        "expect_source_manual_contains": None,
        "expect_page": None,
        "description": "Pressure alarm should return calibration info",
    },
    {
        "query": "completely unrelated quantum physics question about black holes",
        "device": None,
        "expect_status": "NOT_FOUND_IN_MANUAL",
        "expect_safety": False,
        "expect_keywords": [],
        "reject_keywords": [],
        "expect_source_manual_contains": None,
        "expect_page": None,
        "description": "Irrelevant query should return NOT_FOUND",
    },
]

# =============================================================================
# OUTPUT FORMAT CHECKS
# =============================================================================
FORMAT_CHECKS = [
    ("no_think_tags", "Answer should not contain raw <think> tags",
     lambda r: "<think>" not in r["answer"] and "</think>" not in r["answer"]),

    ("no_raw_json_wrapper", "Answer should not be raw JSON starting with {",
     lambda r: not r["answer"].strip().startswith("{")),

    ("has_answer", "Answer should not be empty",
     lambda r: len(r["answer"].strip()) > 10),

    ("checklist_is_list", "Checklist should be a list",
     lambda r: isinstance(r.get("checklist"), list)),

    ("sources_is_list", "Sources should be a list",
     lambda r: isinstance(r.get("sources"), list)),

    ("speech_text_if_found", "speech_text should exist when status is FOUND_IN_MANUAL",
     lambda r: r["status"] != "FOUND_IN_MANUAL" or (r.get("speech_text") and len(r["speech_text"]) > 5)),

    ("safety_body_if_flagged", "safety_body should exist when has_high_priority_safety is True",
     lambda r: not r.get("has_high_priority_safety") or (r.get("safety_body") and len(r["safety_body"]) > 5)),
]


# =============================================================================
# Query Helper
# =============================================================================
def query_backend(query, device=None, top_k=5):
    body = {"query": query, "top_k": top_k}
    if device:
        body["device_name"] = device
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except Exception as e:
        return None, str(e)


# =============================================================================
# Scoring Functions
# =============================================================================
def evaluate_test(test, response):
    """Evaluate a single test case. Returns dict of check_name -> (pass, detail)."""
    results = {}

    # 1. Status check
    if test["expect_status"]:
        passed = response["status"] == test["expect_status"]
        results["status_match"] = (passed,
            f"Expected {test['expect_status']}, got {response['status']}")

    # 2. Safety flag check
    if test["expect_safety"] is not None:
        actual = response.get("has_high_priority_safety", False) or False
        passed = actual == test["expect_safety"]
        results["safety_flag"] = (passed,
            f"Expected safety={test['expect_safety']}, got {actual}")

    # 3. Keyword presence
    answer_lower = response["answer"].lower()
    for kw in test["expect_keywords"]:
        passed = kw.lower() in answer_lower
        results[f"keyword_{kw}"] = (passed,
            f"Keyword '{kw}' {'found' if passed else 'NOT found'} in answer")

    # 4. Reject keywords
    for kw in test["reject_keywords"]:
        passed = kw.lower() not in answer_lower
        results[f"reject_{kw}"] = (passed,
            f"Rejected keyword '{kw}' {'not found (good)' if passed else 'FOUND (bad)'}")

    # 5. Source manual check
    if test["expect_source_manual_contains"]:
        sources = response.get("sources", [])
        if sources:
            manual_name = sources[0].get("manual", "")
            passed = test["expect_source_manual_contains"].lower() in manual_name.lower()
            results["source_manual"] = (passed,
                f"Expected manual containing '{test['expect_source_manual_contains']}', got '{manual_name}'")
        else:
            results["source_manual"] = (False, "No sources returned")

    # 6. Page check
    if test["expect_page"] is not None:
        sources = response.get("sources", [])
        if sources:
            actual_page = sources[0].get("page")
            passed = str(actual_page) == str(test["expect_page"])
            results["source_page"] = (passed,
                f"Expected page {test['expect_page']}, got {actual_page}")
        else:
            results["source_page"] = (False, "No sources returned")

    # 7. Format checks
    for check_name, check_desc, check_fn in FORMAT_CHECKS:
        try:
            passed = check_fn(response)
        except Exception:
            passed = False
        results[check_name] = (passed, check_desc)

    return results


# =============================================================================
# Main Evaluation Runner
# =============================================================================
def run_evaluation():
    print("=" * 70)
    print("  FIXORA RAG EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    total_checks = 0
    total_passed = 0
    test_results = []
    all_responses = []

    for i, test in enumerate(GOLDEN_TESTS, 1):
        print(f"Test {i}/{len(GOLDEN_TESTS)}: {test['description']}")
        print(f"  Query: \"{test['query']}\"  Device: {test['device'] or 'Any'}")

        start = time.time()
        response, error = query_backend(test["query"], test["device"])
        elapsed = time.time() - start

        if error:
            print(f"  ERROR: {error}")
            test_results.append({"test": test, "error": error, "checks": {}})
            continue

        all_responses.append(response)
        checks = evaluate_test(test, response)

        passed = sum(1 for v, _ in checks.values() if v)
        failed = sum(1 for v, _ in checks.values() if not v)
        total_checks += len(checks)
        total_passed += passed

        status_icon = "PASS" if failed == 0 else "FAIL"
        print(f"  Result: [{status_icon}] {passed}/{len(checks)} checks passed ({elapsed:.1f}s)")

        if failed > 0:
            for name, (v, detail) in checks.items():
                if not v:
                    print(f"    FAIL {name}: {detail}")

        test_results.append({"test": test, "checks": checks, "response_preview": response["answer"][:150]})
        print()

        # Small delay to avoid rate limiting
        time.sleep(1.5)

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    # Per-category scores
    categories = {
        "Retrieval": ["status_match", "source_manual", "source_page"],
        "Answer Quality": [f"keyword_{kw}" for t in GOLDEN_TESTS for kw in t["expect_keywords"]] + [f"reject_{kw}" for t in GOLDEN_TESTS for kw in t["reject_keywords"]],
        "Safety Detection": ["safety_flag"],
        "Output Format": [c[0] for c in FORMAT_CHECKS],
    }

    for cat_name, check_names in categories.items():
        cat_total = 0
        cat_passed = 0
        for tr in test_results:
            for name, (v, _) in tr.get("checks", {}).items():
                if name in check_names or any(name.startswith(cn.split("_")[0] + "_") for cn in check_names):
                    cat_total += 1
                    if v:
                        cat_passed += 1
        if cat_total > 0:
            pct = cat_passed / cat_total * 100
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"  {cat_name:20s}  [{bar}]  {cat_passed}/{cat_total} ({pct:.0f}%)")

    print()
    overall_pct = total_passed / total_checks * 100 if total_checks > 0 else 0
    print(f"  OVERALL SCORE: {total_passed}/{total_checks} checks passed ({overall_pct:.0f}%)")
    print()

    # Latency stats
    print("  LATENCY STATS:")
    latencies = []
    for tr in test_results:
        if "error" not in tr:
            latencies.append(0)  # placeholder since we didn't store individual latencies
    print()

    # Known issues detected
    print("  ISSUES DETECTED:")
    issues = set()
    for tr in test_results:
        for name, (v, detail) in tr.get("checks", {}).items():
            if not v:
                if "think" in name:
                    issues.add("LLM leaks <think> tags into answer text")
                elif name == "no_raw_json_wrapper":
                    issues.add("LLM returns raw JSON instead of formatted answer")
                elif name.startswith("source_manual"):
                    issues.add("Wrong manual retrieved for device query")
                elif name == "safety_flag":
                    issues.add("Safety flag mismatch (false positive or false negative)")
                elif name.startswith("keyword"):
                    issues.add("Answer missing expected keywords")
                elif name == "speech_text_if_found":
                    issues.add("Missing speech_text for FOUND_IN_MANUAL responses")

    if issues:
        for issue in sorted(issues):
            print(f"    - {issue}")
    else:
        print("    None detected!")

    print()
    print("=" * 70)

    # Save detailed results to JSON
    report_path = "evaluation_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(GOLDEN_TESTS),
        "total_checks": total_checks,
        "total_passed": total_passed,
        "overall_score_pct": round(overall_pct, 1),
        "tests": [],
    }
    for tr in test_results:
        test_entry = {
            "query": tr["test"]["query"],
            "device": tr["test"]["device"],
            "description": tr["test"]["description"],
            "checks": {k: {"passed": v, "detail": d} for k, (v, d) in tr.get("checks", {}).items()},
        }
        if "response_preview" in tr:
            test_entry["answer_preview"] = tr["response_preview"]
        if "error" in tr:
            test_entry["error"] = tr["error"]
        report["tests"].append(test_entry)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Detailed report saved to: {report_path}")
    print()


if __name__ == "__main__":
    run_evaluation()

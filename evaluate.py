import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000/chat"

# question, expected operation, answer grading criteria
CASES = [
    (
        "Which documents are required for a personal loan?",
        "policy",
        "Lists national ID, income proof for the last three months, "
        "and residential address proof. Cites personal_loan.md chunk 1.",
    ),
    (
        "Does motor insurance cover mechanical breakdown?",
        "policy",
        "Says mechanical breakdown is excluded. "
        "Cites motor_insurance.md chunk 1.",
    ),
    (
        "How quickly will my complaint be acknowledged and resolved?",
        "policy",
        "Acknowledgment: two business days. Resolution: target of ten "
        "business days, with complex cases potentially taking longer. "
        "Cites customer_support.md chunks 2 and 3 appropriately.",
    ),
    (
        "What is the personal loan interest rate?",
        "policy",
        "Says the rate is not specified. Does not invent a rate. "
        "Any citation must point to personal_loan.md chunk 2.",
    ),
    (
        "Calculate payments for 100000 at an annual interest rate "
        "of 12% over 12 months.",
        "calculate",
        "Uses 100000, 12% annually, and 12 months. Monthly payment "
        "8884.88, total repayment 106618.55, total interest 6618.55. "
        "States illustrative assumptions.",
    ),
    (
        "Calculate payments for 12000 at an annual interest rate "
        "of 0% over 12 months.",
        "calculate",
        "Monthly payment 1000.00, total repayment 12000.00, "
        "total interest 0.00. Does not treat zero as missing.",
    ),
    (
        "Calculate monthly payments for a loan of 100000.",
        "calculate",
        "Requests annual interest rate and repayment period. "
        "Does not invent inputs or produce a payment amount.",
    ),
    (
        "What repayment periods does the personal loan policy allow? "
        "Also calculate payments for 100000 at an annual interest rate "
        "of 12% over 24 months.",
        "both",
        "States 12 to 60 months with personal_loan.md chunk 2 citation. "
        "Calculation uses 100000, 12% annually, and 24 months: "
        "monthly 4707.35, total 112976.33, interest 12976.33. "
        "Does not describe 12% as an official policy rate.",
    ),
]


def ask_api(question):
    request = Request(
        API_URL,
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = perf_counter()

    with urlopen(request, timeout=120) as response:
        body = json.load(response)

    return body, round(perf_counter() - started, 3)


def main():
    # Stop early if the local server is not available.
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            health = json.load(response)
        if health.get("status") != "ok":
            raise ValueError("Health check did not return ok.")
    except Exception as error:
        print(f"Server check failed: {error}")
        print("Start Uvicorn in another terminal, then rerun this script.")
        return

    folder = BASE_DIR / "evaluation_results"
    folder.mkdir(exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = folder / f"evaluation_{stamp}.json"

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set": "development_v1",
        "planned_cases": len(CASES),
        "grading": "Human answer review plus automatic route check",
        "results": [],
    }

    def save():
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    save()

    for number, (question, expected_route, criteria) in enumerate(CASES, 1):
        print(f"\nCASE {number}/{len(CASES)}")
        print(f"Question: {question}")
        print("Waiting for the API...")

        row = {
            "id": number,
            "question": question,
            "expected_operation": expected_route,
            "criteria": criteria,
            "request_ok": False,
            "route_pass": False,
            "answer_pass": False,
            "overall_pass": False,
        }

        try:
            body, elapsed = ask_api(question)

            answer = body["answer"]
            operation = body["operation"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("API returned an empty or invalid answer.")

            row.update({
                "request_ok": True,
                "response": body,
                "client_elapsed_seconds": elapsed,
                "route_pass": operation == expected_route,
                "answer_pass": None,
            })

        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            report["results"].append(row)
            save()
            print(f"Request failed: {row['error']}")
            print("Recorded as failed. Stopping to avoid cascading failures.")
            print(f"Partial report saved to: {output}")
            return

        # Save the response before asking for review.
        report["results"].append(row)
        save()

        print(f"\nActual operation: {operation}")
        print(f"Expected operation: {expected_route}")
        print(f"\nAnswer:\n{answer}")
        print(f"\nPASS CRITERIA:\n{criteria}")
        print("Also reject unsupported claims or incorrect citations.")

        while True:
            grade = input("\nDoes the answer meet ALL criteria? [y/n]: ")
            grade = grade.strip().lower()
            if grade in {"y", "n"}:
                break

        row["answer_pass"] = grade == "y"
        row["overall_pass"] = row["route_pass"] and row["answer_pass"]

        if not row["answer_pass"]:
            row["review_note"] = input("Briefly describe the problem: ").strip()

        save()

    results = report["results"]
    timings = [row["client_elapsed_seconds"] for row in results]

    report["summary"] = {
        "completed_cases": len(results),
        "route_passes": sum(row["route_pass"] for row in results),
        "human_answer_passes": sum(row["answer_pass"] for row in results),
        "overall_passes": sum(row["overall_pass"] for row in results),
        "overall_pass_rate_percent": round(
            100 * sum(row["overall_pass"] for row in results) / len(results), 1
        ),
        "mean_client_seconds": round(mean(timings), 2),
        "median_client_seconds": round(median(timings), 2),
        "max_client_seconds": round(max(timings), 2),
    }

    save()
    print("\nEVALUATION SUMMARY")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report saved to: {output}")


if __name__ == "__main__":
    main()
import math


def calculate_installment(
    principal: float,
    annual_rate_percent: float,
    months: int,
) -> dict:
    """Calculate a fixed-rate loan with equal monthly payments."""

    # Reject invalid inputs before doing any calculations.
    for name, value in (
        ("principal", principal),
        ("annual_rate_percent", annual_rate_percent),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

    if principal <= 0:
        raise ValueError("The loan amount must be greater than zero.")

    if annual_rate_percent < 0:
        raise ValueError("The annual interest rate cannot be negative.")

    if isinstance(months, bool) or not isinstance(months, int):
        raise ValueError("The repayment period must be a whole number of months.")

    if months <= 0:
        raise ValueError("The repayment period must be greater than zero.")

    # Convert an annual percentage into a monthly decimal rate.
    monthly_rate = annual_rate_percent / 100 / 12

    if monthly_rate == 0:
        monthly_payment = principal / months
    else:
        # Numerically stable version of the standard amortization formula.
        denominator = -math.expm1(
            -months * math.log1p(monthly_rate)
        )
        monthly_payment = principal * monthly_rate / denominator

    total_repayment = monthly_payment * months
    total_interest = total_repayment - principal

    if not all(
        math.isfinite(value)
        for value in (monthly_payment, total_repayment, total_interest)
    ):
        raise ValueError("These inputs are too large to calculate.")

    # Keep full precision during calculations; round only the results.
    return {
        "principal": round(principal, 2),
        "annual_rate_percent": annual_rate_percent,
        "months": months,
        "monthly_payment": round(monthly_payment, 2),
        "total_repayment": round(total_repayment, 2),
        "total_interest": round(total_interest, 2),
        "assumptions": (
            "Fixed nominal annual interest rate, equal monthly payments "
            "at the end of each month, and no fees. "
            "Illustrative estimate using a user-supplied rate. "
            "Actual lender rounding may differ."
        ),
    }


def main():
    print("Demo loan installment calculator")
    print("Supply your own example interest rate, not a policy rate.\n")

    try:
        principal = float(input("Loan amount: "))
        annual_rate = float(input("Annual interest rate (%): "))
        months = int(input("Repayment period in months: "))

        result = calculate_installment(principal, annual_rate, months)

    except ValueError as error:
        print(f"\nInput error: {error}")
        return

    print(f"\nMonthly payment: {result['monthly_payment']:,.2f}")
    print(f"Total repayment: {result['total_repayment']:,.2f}")
    print(f"Total interest: {result['total_interest']:,.2f}")
    print(f"\nAssumptions: {result['assumptions']}")


if __name__ == "__main__":
    main()
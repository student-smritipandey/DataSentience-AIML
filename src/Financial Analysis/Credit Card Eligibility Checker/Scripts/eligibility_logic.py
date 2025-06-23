def check_eligibility(age, income, employment_status):
    if age < 18:
        return False, "Age must be 18 or above."
    if income < 20000:
        return False, "Monthly income must be ₹20,000 or more."
    if employment_status.lower() not in ['salaried', 'self-employed']:
        return False, "Only salaried or self-employed individuals are eligible."
    return True, "You are eligible for a credit card."

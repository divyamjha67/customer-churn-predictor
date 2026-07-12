import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers to generate
num_customers = 1500

# Helper to generate unique Customer IDs
def generate_customer_ids(n):
    ids = []
    for _ in range(n):
        num_part = np.random.randint(1000, 9999)
        letter_part = "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5))
        ids.append(f"{num_part}-{letter_part}")
    return ids

customer_ids = generate_customer_ids(num_customers)
genders = np.random.choice(["Male", "Female"], num_customers)
senior_citizens = np.random.choice([0, 1], num_customers, p=[0.85, 0.15])
partners = np.random.choice(["Yes", "No"], num_customers)
dependents = np.random.choice(["Yes", "No"], num_customers, p=[0.7, 0.3])

# Tenure in months (1 to 72)
tenures = np.random.randint(1, 73, num_customers)

# Phone Service
phone_services = np.random.choice(["Yes", "No"], num_customers, p=[0.9, 0.1])

# Multiple Lines
multiple_lines = []
for i in range(num_customers):
    if phone_services[i] == "No":
        multiple_lines.append("No phone service")
    else:
        multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))

# Internet Service
internet_services = np.random.choice(["Fiber optic", "DSL", "No"], num_customers, p=[0.45, 0.35, 0.20])

# Internet-related services
online_securities = []
online_backups = []
device_protections = []
tech_supports = []
streaming_tvs = []
streaming_movies = []

for i in range(num_customers):
    if internet_services[i] == "No":
        online_securities.append("No internet service")
        online_backups.append("No internet service")
        device_protections.append("No internet service")
        tech_supports.append("No internet service")
        streaming_tvs.append("No internet service")
        streaming_movies.append("No internet service")
    else:
        online_securities.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
        online_backups.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
        device_protections.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
        tech_supports.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
        streaming_tvs.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
        streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))

# Contracts (correlated with tenure)
contracts = []
for i in range(num_customers):
    t = tenures[i]
    if t < 12:
        contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.85, 0.12, 0.03]))
    elif t < 36:
        contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.35, 0.45, 0.20]))
    else:
        contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.08, 0.22, 0.70]))

paperless_billings = np.random.choice(["Yes", "No"], num_customers, p=[0.60, 0.40])
payment_methods = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    num_customers,
    p=[0.35, 0.25, 0.20, 0.20]
)

# Calculate MonthlyCharges
monthly_charges = []
for i in range(num_customers):
    if internet_services[i] == "No":
        # Dial-up or no internet, cheap
        base = 20.0 + np.random.uniform(-3, 4)
        if phone_services[i] == "Yes":
            base += 5.0
        monthly_charges.append(round(base, 2))
    elif internet_services[i] == "DSL":
        base = 45.0 + np.random.uniform(-5, 5)
        if phone_services[i] == "Yes": base += 5.0
        if online_securities[i] == "Yes": base += 6.0
        if online_backups[i] == "Yes": base += 6.0
        if device_protections[i] == "Yes": base += 6.0
        if tech_supports[i] == "Yes": base += 7.0
        if streaming_tvs[i] == "Yes": base += 10.0
        if streaming_movies[i] == "Yes": base += 10.0
        monthly_charges.append(round(base, 2))
    else: # Fiber optic
        base = 75.0 + np.random.uniform(-5, 8)
        if phone_services[i] == "Yes": base += 5.0
        if online_securities[i] == "Yes": base += 8.0
        if online_backups[i] == "Yes": base += 8.0
        if device_protections[i] == "Yes": base += 8.0
        if tech_supports[i] == "Yes": base += 9.0
        if streaming_tvs[i] == "Yes": base += 12.0
        if streaming_movies[i] == "Yes": base += 12.0
        monthly_charges.append(round(base, 2))

# Total charges = tenure * monthly_charges + some noise
total_charges = []
for i in range(num_customers):
    t = tenures[i]
    mc = monthly_charges[i]
    noise = np.random.normal(1.0, 0.02) # +/- 2% noise
    tc = t * mc * noise
    total_charges.append(round(max(mc, tc), 2))

# Churn logic (calculating probability and sampling)
churn = []
for i in range(num_customers):
    p = 0.12 # baseline
    
    if contracts[i] == "Month-to-month":
        p += 0.38
    elif contracts[i] == "One year":
        p += 0.05
    elif contracts[i] == "Two year":
        p -= 0.12
        
    if internet_services[i] == "Fiber optic":
        p += 0.15
        
    if online_securities[i] == "No":
        p += 0.08
    if tech_supports[i] == "No":
        p += 0.08
        
    if payment_methods[i] == "Electronic check":
        p += 0.12
        
    if paperless_billings[i] == "Yes":
        p += 0.04
        
    if tenures[i] < 12:
        p += 0.18
    elif tenures[i] > 36:
        p -= 0.10
    elif tenures[i] > 60:
        p -= 0.20
        
    if monthly_charges[i] > 80:
        p += 0.10
        
    if senior_citizens[i] == 1:
        p += 0.05
        
    p = np.clip(p, 0.01, 0.95)
    churn.append(np.random.choice(["Yes", "No"], p=[p, 1 - p]))

# Create DataFrame
df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": genders,
    "SeniorCitizen": senior_citizens,
    "Partner": partners,
    "Dependents": dependents,
    "tenure": tenures,
    "PhoneService": phone_services,
    "MultipleLines": multiple_lines,
    "InternetService": internet_services,
    "OnlineSecurity": online_securities,
    "OnlineBackup": online_backups,
    "DeviceProtection": device_protections,
    "TechSupport": tech_supports,
    "StreamingTV": streaming_tvs,
    "StreamingMovies": streaming_movies,
    "Contract": contracts,
    "PaperlessBilling": paperless_billings,
    "PaymentMethod": payment_methods,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn
})

# Make a small fraction of TotalCharges empty strings to simulate real-world data preprocessing needs
df["TotalCharges"] = df["TotalCharges"].astype(object)
empty_indices = np.random.choice(num_customers, 5, replace=False)
for idx in empty_indices:
    df.loc[idx, "TotalCharges"] = " "

# Ensure output folder exists
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "customer_churn_data.csv")
df.to_csv(output_path, index=False)
print(f"Generated synthetic customer churn data successfully in {output_path} ({num_customers} rows)")

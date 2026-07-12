// API URL
const API_URL = "";

// Initialize page
document.addEventListener("DOMContentLoaded", () => {
    fetchMetrics();
    adjustCharges();
    toggleInternetServices();
    togglePhoneServices();
});

// Switch Tab
function switchTab(tabName) {
    // Toggles view
    document.querySelectorAll(".tab-view").forEach(view => {
        view.classList.remove("active");
    });
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.remove("active");
    });

    document.getElementById(`${tabName}-view`).classList.add("active");
    document.getElementById(`${tabName}-tab-btn`).classList.add("active");
}

// Fetch ML Metrics on Load
async function fetchMetrics() {
    try {
        const response = await fetch(`${API_URL}/api/metrics`);
        if (response.ok) {
            const data = await response.json();
            
            // Format to percentage
            const formatPct = (val) => (val * 100).toFixed(1) + "%";
            
            document.getElementById("kpi-accuracy").innerText = formatPct(data.metrics.accuracy);
            document.getElementById("kpi-auc").innerText = data.metrics.roc_auc.toFixed(3);
        }
    } catch (err) {
        console.error("Error fetching metrics:", err);
    }
}

// Form Helpers: Enable/Disable Internet Addons
function toggleInternetServices() {
    const internetService = document.getElementById("internet-addon-service") || document.getElementById("InternetService").value;
    const isNoInternet = internetService === "No";
    
    const addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];
    
    addons.forEach(addonId => {
        const select = document.getElementById(addonId);
        if (isNoInternet) {
            select.value = "No internet service";
            select.disabled = true;
        } else {
            if (select.value === "No internet service") {
                select.value = "No";
            }
            select.disabled = false;
        }
    });
}

// Form Helpers: Enable/Disable Phone Addons
function togglePhoneServices() {
    const phoneService = document.getElementById("PhoneService").value;
    const isNoPhone = phoneService === "No";
    const multipleLinesSelect = document.getElementById("MultipleLines");
    
    if (isNoPhone) {
        multipleLinesSelect.value = "No phone service";
        multipleLinesSelect.disabled = true;
    } else {
        if (multipleLinesSelect.value === "No phone service") {
            multipleLinesSelect.value = "No";
        }
        multipleLinesSelect.disabled = false;
    }
}

// Automatically calculate realistic Monthly Charges based on features
function adjustCharges() {
    const internetService = document.getElementById("InternetService").value;
    const phoneService = document.getElementById("PhoneService").value;
    const contract = document.getElementById("Contract").value;
    
    let base = 0;
    
    if (internetService === "No") {
        base = 20.0;
        if (phoneService === "Yes") base += 5.0;
    } else if (internetService === "DSL") {
        base = 45.0;
        if (phoneService === "Yes") base += 5.0;
        
        // Addons
        if (document.getElementById("OnlineSecurity").value === "Yes") base += 6.0;
        if (document.getElementById("OnlineBackup").value === "Yes") base += 6.0;
        if (document.getElementById("DeviceProtection").value === "Yes") base += 6.0;
        if (document.getElementById("TechSupport").value === "Yes") base += 7.0;
        if (document.getElementById("StreamingTV").value === "Yes") base += 10.0;
        if (document.getElementById("StreamingMovies").value === "Yes") base += 10.0;
    } else { // Fiber optic
        base = 75.0;
        if (phoneService === "Yes") base += 5.0;
        
        // Addons
        if (document.getElementById("OnlineSecurity").value === "Yes") base += 8.0;
        if (document.getElementById("OnlineBackup").value === "Yes") base += 8.0;
        if (document.getElementById("DeviceProtection").value === "Yes") base += 8.0;
        if (document.getElementById("TechSupport").value === "Yes") base += 9.0;
        if (document.getElementById("StreamingTV").value === "Yes") base += 12.0;
        if (document.getElementById("StreamingMovies").value === "Yes") base += 12.0;
    }
    
    // Contract discount
    if (contract === "One year") {
        base *= 0.90; // 10% off
    } else if (contract === "Two year") {
        base *= 0.80; // 20% off
    }
    
    document.getElementById("MonthlyCharges").value = base.toFixed(2);
    recalculateTotalCharges();
}

// Recalculate Total Charges based on monthly cost and tenure
function recalculateTotalCharges() {
    const tenure = parseInt(document.getElementById("tenure").value) || 1;
    const monthlyCharges = parseFloat(document.getElementById("MonthlyCharges").value) || 0;
    
    const estTotal = tenure * monthlyCharges;
    document.getElementById("TotalCharges").value = estTotal.toFixed(2);
}

// Execute Prediction Call
async function getPrediction() {
    // Show loader
    document.getElementById("result-placeholder").classList.add("hidden");
    document.getElementById("result-display").classList.add("hidden");
    document.getElementById("result-loader").classList.remove("hidden");
    
    // Read Form Fields
    const payload = {
        gender: document.getElementById("gender").value,
        SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        tenure: parseInt(document.getElementById("tenure").value),
        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,
        InternetService: document.getElementById("InternetService").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        TechSupport: document.getElementById("TechSupport").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        Contract: document.getElementById("Contract").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
        TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
    };
    
    try {
        const response = await fetch(`${API_URL}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server error occurred during prediction.");
        }
        
        const result = await response.json();
        
        // Hide loader, show results
        document.getElementById("result-loader").classList.add("hidden");
        document.getElementById("result-display").classList.remove("hidden");
        
        // Update Risk Gauge
        const circle = document.getElementById("risk-gauge");
        const proba = result.probability;
        const percent = Math.round(proba * 100);
        
        // Circumference is 471
        const offset = 471 - (proba * 471);
        circle.style.strokeDashoffset = offset;
        
        // Update Gauge Stroke Color & Risk Badge Styling
        const badge = document.getElementById("risk-badge-element");
        badge.innerText = result.risk_level;
        badge.className = "risk-badge"; // Reset classes
        
        if (result.risk_level === "Low Risk") {
            circle.setAttribute("stroke", "url(#cyanGlow)");
            badge.classList.add("low");
        } else if (result.risk_level === "Medium Risk") {
            circle.setAttribute("stroke", "url(#yellowGlow)");
            badge.classList.add("medium");
        } else {
            circle.setAttribute("stroke", "url(#redGlow)");
            badge.classList.add("high");
        }
        
        document.getElementById("risk-proba").innerText = `${percent}%`;
        
        // Update Key Factors list
        const factorsList = document.getElementById("risk-factors-list");
        factorsList.innerHTML = "";
        result.key_factors.forEach(factor => {
            const li = document.createElement("li");
            li.className = "insight-item risk";
            li.innerText = factor;
            factorsList.appendChild(li);
        });
        
        // Update Mitigations list
        const mitigationsList = document.getElementById("mitigations-list");
        mitigationsList.innerHTML = "";
        result.recommendations.forEach(rec => {
            const li = document.createElement("li");
            li.className = "insight-item rec";
            li.innerText = rec;
            mitigationsList.appendChild(li);
        });
        
    } catch (err) {
        // Reset state, display error
        document.getElementById("result-loader").classList.add("hidden");
        document.getElementById("result-placeholder").classList.remove("hidden");
        alert(`Failed to retrieve prediction: ${err.message}`);
    }
}

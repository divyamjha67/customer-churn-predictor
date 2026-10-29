"""One input contract for training, JSON requests and CSV uploads."""
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from typing import Literal

NUMERIC = {"tenure": (0, 1200), "MonthlyCharges": (0, 1000), "TotalCharges": (0, 1000000)}
YES_NO = ["Yes", "No"]
INTERNET_SERVICE = ["Yes", "No", "No internet service"]
CATEGORIES = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_SERVICE,
    "TechSupport": INTERNET_SERVICE,
    "OnlineBackup": INTERNET_SERVICE,
    "DeviceProtection": INTERNET_SERVICE,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    "PaperlessBilling": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "StreamingTV": INTERNET_SERVICE,
    "StreamingMovies": INTERNET_SERVICE,
}
FEATURES = list(NUMERIC) + list(CATEGORIES)

class InputBase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("tenure", "MonthlyCharges", "TotalCharges", mode="before", check_fields=False)
    @classmethod
    def numeric_not_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("Use a number, not a boolean")
        return value

    @model_validator(mode="after")
    def consistent_services(self):
        for field in ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]:
            absent = getattr(self, field) == "No internet service"
            if absent != (self.InternetService == "No"):
                raise ValueError(f"{field} must agree with InternetService")
        if (self.MultipleLines == "No phone service") != (self.PhoneService == "No"):
            raise ValueError("MultipleLines must agree with PhoneService")
        return self

    @field_validator("tenure", check_fields=False)
    @classmethod
    def whole_months(cls, value):
        if value != int(value):
            raise ValueError("tenure must be a whole number of months")
        return value

    @field_validator("TotalCharges", mode="before", check_fields=False)
    @classmethod
    def blank_total(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

Customer = create_model(
    "Customer", __base__=InputBase,
    **{name: ((float | None) if name == "TotalCharges" else float,
              Field(default=None, ge=bounds[0], le=bounds[1]) if name == "TotalCharges"
              else Field(..., ge=bounds[0], le=bounds[1])) for name, bounds in NUMERIC.items()},
    **{name: (Literal[tuple(options)], ...) for name, options in CATEGORIES.items()},
)

def frame(records):
    """Column order is fixed; optional total charges use the fitted imputer."""
    result = pd.DataFrame(records, columns=FEATURES)
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    return result

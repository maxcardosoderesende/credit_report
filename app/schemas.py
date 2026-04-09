# Defining strict rules for what the API accepts as input. 

# In n a controlled Jupyter notebook, you know exactly what the data looks like. In production, you have zero control. Someone might send:

    # Text where a number is expected ("age": "forty")
    # Negative values where only positive makes sense ("income": -5000)
    # Missing required fields
    # Values that are technically valid but make no business sense ("age": 200)
    # Pydantic lets us define a schema — a set of rules — for our input data. FastAPI uses this schema to automatically validate every incoming request.

    # If the data doesn’t match, the API returns a detailed error message explaining exactly what’s wrong. 

# to create the schemas, we use pydantic (FASTApi)

from pydantic import BaseModel, Field, field_validator


# =========================
# Input Schema
# =========================
class LoanApplication(BaseModel):
    """
    Schema defining the structure and validation rules
    for incoming loan application data.
    """

    # External credit scores (normalized)
    ext_source_1: float = Field(..., ge=0.0, le=1.0)
    ext_source_2: float = Field(..., ge=0.0, le=1.0)
    ext_source_3: float = Field(..., ge=0.0, le=1.0)
    # What happens if someone sends ext_source_1: 5.0? FastAPI catches it and returns:
    # {"detail": [{"msg": "Input should be less than or equal to 1"}]} -
    # with HTTP status 422 (Unprocessable Entity). 

    # Financial fields
    amt_income_total: float = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    amt_annuity: float = Field(..., gt=0)
    amt_goods_price: float = Field(..., gt=0)

    # Personal info
    code_gender: int = Field(..., ge=0, le=1)
    flag_own_car: int = Field(..., ge=0, le=1)
    flag_own_realty: int = Field(..., ge=0, le=1)
    cnt_children: int = Field(..., ge=0, le=20)

    # Derived / business logic fields
    age_years: float = Field(..., ge=18, le=80)
    years_employed: float = Field(..., ge=0, le=50)
    years_id_publish: float = Field(..., ge=0, le=60)
    education_level: int = Field(..., ge=0, le=4)


    # =========================
    # Custom validation ;  Business Rules
    # =========================
    @field_validator("amt_credit")
    @classmethod
    def credit_must_be_reasonable(cls, v, info):
        """
        Validate that credit is not unrealistically high compared to income.
        """
        income = info.data.get("amt_income_total")

        if income is not None:
            ratio = v / (income + 1)  # avoid division by zero

            if ratio > 100:
                raise ValueError(
                    f"Credit-to-income ratio ({ratio:.0f}x) seems unrealistic"
                )

        return v


# =========================
# Output Schema
# =========================
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=No Default, 1=Default")
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    risk_category: str = Field(..., description="Low, Medium, High")


# =========================
# Health Check Schema
# =========================
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

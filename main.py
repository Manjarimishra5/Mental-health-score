import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal


# ==================================================
# LOAD MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

model = joblib.load(MODEL_PATH)


# Countries used during model training
top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# ==================================================
# FASTAPI APP
# ==================================================

app = FastAPI(
    title="Mental Health Score API",
    description="Predicts student mental health score from social media usage and lifestyle data.",
    version="1.0"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# INPUT MODEL
# ==================================================

class StudentData(BaseModel):

    age: int = Field(
        ...,
        ge=10,
        le=100
    )

    gender: Literal[
        "Male",
        "Female"
    ]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    daily_unlocks: int = Field(
        ...,
        ge=0
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )

    stress_level: Literal[
        "Medium",
        "Low",
        "Very High",
        "High"
    ]


# ==================================================
# OUTPUT MODEL
# ==================================================

class PredictionResponse(BaseModel):

    predicted_mental_health_score: float


# ==================================================
# HOME ROUTE
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Mental Health Score API is running"
    }


# ==================================================
# PREDICTION ROUTE
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: StudentData):

    # ----------------------------------------------
    # Convert Pydantic object to dictionary
    # ----------------------------------------------

    input_data = data.model_dump()


    # ----------------------------------------------
    # Group countries
    # ----------------------------------------------

    if input_data["country"] not in top_countries:
        input_data["country"] = "Other"


    # ----------------------------------------------
    # Rename country to match training dataset
    # ----------------------------------------------

    input_data["Grouped_country"] = input_data.pop("country")


    # ----------------------------------------------
    # Create input DataFrame
    # ----------------------------------------------

    input_row = pd.DataFrame([
        {
            "Age": input_data["age"],

            "Gender": input_data["gender"],

            "Grouped_country": input_data["Grouped_country"],

            "Academic_Level": input_data["academic_level"],

            "Most_Used_Platform": input_data["most_used_platform"],

            "Purpose_Of_Use": input_data["purpose_of_use"],

            "Avg_Daily_Usage_Hours": input_data[
                "avg_daily_usage_hours"
            ],

            "Daily_Unlocks": input_data[
                "daily_unlocks"
            ],

            "Study_Hours": input_data[
                "study_hours"
            ],

            "Physical_Activity_Hours": input_data[
                "physical_activity_hours"
            ],

            "Sleep_Hours_Per_Night": input_data[
                "sleep_hours_per_night"
            ],

            "Stress_Level": input_data[
                "stress_level"
            ]
        }
    ])


    # ----------------------------------------------
    # Make prediction
    # ----------------------------------------------
    print("INPUT COLUMNS:")
    print(input_row.columns.tolist())

    print("MODEL COLUMNS:")
    print(model.feature_names_in_.tolist())

    prediction = model.predict(input_row)[0]


    # ----------------------------------------------
    # Return prediction
    # ----------------------------------------------

    return PredictionResponse(
        predicted_mental_health_score=round(
            float(prediction),
            2
        )
    )
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq

# 1. Load your .env configuration
load_dotenv()

app = FastAPI(title="AI Menstrual Cycle Tracker Backend")

# 2. Add CORS Middleware to allow your index.html file to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],  # Allows all communication headers
)

# 3. Check for the Groq API Key
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("Missing GROQ_API_KEY inside your .env configuration.")

# Initialize the Groq Engine Client
groq_client = Groq(api_key=groq_key)

# 4. Data Structure Models for Request Validations
class CycleLog(BaseModel):
    start_date: str          # Format: YYYY-MM-DD
    end_date: Optional[str]  # Format: YYYY-MM-DD (Optional if current cycle is active)
    flow_intensity: str      # e.g., Light, Medium, Heavy

class TrackerRequest(BaseModel):
    logs: List[CycleLog]     # Chronological log history of the user's previous cycles
    user_notes: Optional[str] = "" # Symptoms, cramps, moods, or general custom info

# 5. API Endpoints
@app.get("/")
def read_root():
    return {"message": "Menstrual Cycle Tracking API is active."}

@app.post("/analyze-cycle")
async def analyze_cycle(data: TrackerRequest):
    if not data.logs:
        raise HTTPException(status_code=400, detail="Cycle log history cannot be empty.")
    
    # Construct the instruction string for the LLM
    formatted_logs = ""
    for idx, log in enumerate(data.logs, 1):
        formatted_logs += f"\nCycle {idx}: Started {log.start_date}, Ended {log.end_date or 'Ongoing'}, Flow: {log.flow_intensity}"

    # System instruction handling calculations and forcing raw JSON keys
    system_prompt = (
        "You are an expert, empathetic medical assistant specializing in reproductive health and menstruation tracking. "
        "Analyze the provided calendar logs to calculate average cycle lengths and accurately forecast upcoming windows. "
        "You must respond ONLY with a single JSON object. Do not include any introductory text, notes, or markdown formatting outside the JSON object. "
        "Your output structure must match these exact keys: "
        "{\n"
        "  \"next_period_prediction\": \"string describing estimated date range\",\n"
        "  \"estimated_ovulation_window\": \"string describing estimated fertile range\",\n"
        "  \"cycle_regularity_status\": \"brief analytical note on regularity\",\n"
        "  \"health_insights\": \"actionable wellness advice based on logs or user notes\"\n"
        "}"
    )
    
    user_prompt = f"Here is my historical calendar tracking data:{formatted_logs}\n\nAdditional personal observations: {data.user_notes}"

    try:
        # Call Groq's high-performance LLM enforcing JSON Object structural modes
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},  # Enforces strict JSON output structural rules
            temperature=0.3,
        )
        
        return {
            "status": "success",
            "analysis": completion.choices[0].message.content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")
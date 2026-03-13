from pydantic import BaseModel

class CompareResponse(BaseModel):
    matchScore: float
    isMatch: bool
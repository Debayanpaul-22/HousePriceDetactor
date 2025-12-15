from pydantic import BaseModel

class HouseInput(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning:str
    parking:int
    prefarea:str
    furnishingstatus:str
    location_type:str
    base_ppsf:int


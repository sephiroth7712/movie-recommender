from pydantic import BaseModel, EmailStr

# User schemas
class UserCreate(BaseModel):
    username: EmailStr  # Ensure it's a valid email format
    password: str        # Plain password (hashed in backend)

class UserResponse(BaseModel):
    id: int
    username: EmailStr

    class Config:
        orm_mode = True


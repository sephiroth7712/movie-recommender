from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db, engine
from models import Base
from schemas import ItemCreate, ItemResponse
from schemas import UserCreate, UserResponse
import crud

app = FastAPI()


# Create tables
@app.on_event("startup")
async def startup():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise e


# # Routes


# Create an item
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_user(db, user)


# Get all items
@app.get("/items/", response_model=list[ItemResponse])
async def read_items(db: AsyncSession = Depends(get_db)):
    return await crud.get_items(db)


# Get a specific item
@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
    item = await crud.get_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


# Update an item
@app.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(
    item_id: int, item: ItemCreate, db: AsyncSession = Depends(get_db)
):
    updated_item = await crud.update_item(db, item_id, item)
    if not updated_item:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated_item


# Delete an item
@app.delete("/items/{item_id}")
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
    deleted_item = await crud.delete_item(db, item_id)
    if not deleted_item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item deleted"}

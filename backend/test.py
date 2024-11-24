import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def test_connection():
    try:
        engine = create_async_engine(
            "postgresql+asyncpg://naruto:{password}@127.0.0.1:5432/mydatabase",
            echo=True
        )
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
            print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {str(e)}")

if __name__ == "_main_":
    asyncio.run(test_connection())
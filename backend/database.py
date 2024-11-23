from sqlalchemy.ext.asyncio import AsyncSession,create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker

from urllib.parse import quote_plus

password = quote_plus("Jignyas@123")
DATABASE_URL = f"postgresql+asyncpg://postgres:{password}@127.0.0.1:5432/mydatabase"

# Create async engine
engine = create_async_engine(DATABASE_URL, future=True, echo=True, poolclass=NullPool)

# Create a sessionmaker for async sessions
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Dependency for FastAPI routes
async def get_db():
    async with async_session() as session:
        yield session

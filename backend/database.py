from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from dotenv import dotenv_values
from urllib.parse import quote_plus

config = dotenv_values(".env")


DATABASE_URL = f"postgresql+asyncpg://{config["DATABASE_USER"]}:{quote_plus(config["DATABASE_PASSWORD"])}@{config["DATABASE_DOMAIN"]}/{config["DATABASE_NAME"]}"

# Create async engine
engine = create_async_engine(DATABASE_URL, future=True, echo=False, poolclass=NullPool)

# Create a sessionmaker for async sessions
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


# Dependency for FastAPI routes
async def get_db():
    async with async_session() as session:
        yield session

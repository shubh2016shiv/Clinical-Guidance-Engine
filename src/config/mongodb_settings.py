"""MongoDB connection and authentication settings."""

from pydantic import BaseModel, Field


class MongoDBSettings(BaseModel):
    """Fields for persistent operational data storage."""

    mongo_host: str = Field(
        default="localhost",
        description="MongoDB server host. Use 'mongodb' inside Docker Compose network.",
    )

    mongo_port: int = Field(
        default=27017,
        ge=1,
        le=65535,
        description="MongoDB server port.",
    )

    mongo_root_user: str = Field(
        default="admin",
        description="MongoDB root username.",
    )

    mongo_root_password: str = Field(
        default="password123",
        description="MongoDB root password.",
    )

    mongo_db_name: str = Field(
        default="asclepius",
        description="MongoDB database name for the application.",
    )

    mongo_auth_source: str = Field(
        default="admin",
        description="MongoDB authentication source database.",
    )

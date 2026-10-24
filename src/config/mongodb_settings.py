"""MongoDB connection and authentication settings.

Big-picture flow:
1. Describe where the MongoDB server is (host, port).
2. Describe the credentials and database name used to connect to it.
"""

from pydantic import BaseModel, Field

# See paths.py for what Pydantic's BaseModel and Field do.


# MongoDBSettings: groups everything needed to connect to MongoDB, the
# database used for persistent operational data (separate from Redis,
# which is a fast cache, and Milvus, which stores search vectors).
class MongoDBSettings(BaseModel):
    """Fields for persistent operational data storage."""

    mongo_host: str = Field(
        default="localhost",
        description="MongoDB server host. Use 'mongodb' inside Docker Compose network.",
    )
    # mongo_host: where to find the MongoDB server. "localhost" when
    # running outside Docker; the service name "mongodb" when running
    # inside the Docker Compose network, since containers address each
    # other by service name instead of "localhost".

    mongo_port: int = Field(
        default=27017,
        ge=1,
        le=65535,
        description="MongoDB server port.",
    )
    # mongo_port: which network port MongoDB is listening on.

    mongo_root_user: str = Field(
        default="admin",
        description="MongoDB root username.",
    )
    # mongo_root_user: the account used to authenticate to MongoDB.

    mongo_root_password: str = Field(
        default="password123",
        description="MongoDB root password.",
    )
    # mongo_root_password: the password for the account above. NOTE: this
    # default is a placeholder meant for local development only - it
    # should always be overridden in any real deployment via the .env file.

    mongo_db_name: str = Field(
        default="asclepius",
        description="MongoDB database name for the application.",
    )
    # mongo_db_name: which database (inside the MongoDB server) this app
    # reads and writes to.

    mongo_auth_source: str = Field(
        default="admin",
        description="MongoDB authentication source database.",
    )
    # mongo_auth_source: which database MongoDB checks the username/password
    # against - often different from the database the app actually uses.

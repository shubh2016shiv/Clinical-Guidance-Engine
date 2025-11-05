# Infrastructure

Docker Compose configuration for the Asclepius Healthcare Chatbot infrastructure services.

## Overview

This infrastructure provides the data storage and retrieval services required for the chatbot application. It includes vector database for semantic search, document database for conversation storage, cache for performance optimization, and supporting services.

## Services

### Milvus (Vector Database)

**Purpose**: Stores and queries high-dimensional vector embeddings for pharmaceutical drug data.

**Why**: Enables semantic similarity search over drug information. The application uses embeddings to find relevant drugs based on natural language queries rather than exact keyword matching.

**Components**:
- **milvus-standalone**: Main vector database service (port 19530)
- **etcd**: Metadata storage and service discovery for Milvus (port 2379)
- **minio**: Object storage for Milvus data files (ports 9000, 9001)
- **attu**: Web-based management interface for Milvus (port 8000)

**Data**: Pharmaceutical drug embeddings stored as float vectors (768 dimensions).

### MongoDB (Document Database)

**Purpose**: Stores conversation history, user sessions, and structured application data.

**Why**: Provides flexible document storage for JSON-like data structures. Suitable for storing conversation records with variable schemas and complex nested data.

**Components**:
- **mongodb**: Document database service (port 27017)
- Authentication: `admin` / `password123`

**Data**: Conversation records, session metadata, application state.

### Redis (Cache)

**Purpose**: In-memory key-value store for caching frequently accessed data and session management.

**Why**: Reduces database load by caching query results and session data. Improves response times for repeated queries.

**Components**:
- **redis**: Cache service (port 6379)
- Authentication: Password `redis123`

**Data**: Cached query results, session tokens, temporary data.

## Prerequisites

- Docker Desktop or Docker Engine 20.10+
- Docker Compose v2.0+ or docker-compose v1.29+
- Python 3.9+ (for management script)
- 8GB+ available disk space for volumes

## Usage

### Start Services

```bash
python manage_infrastructure.py --start
```

Starts all services in detached mode. Checks for conflicting containers from other projects and handles them automatically.

### Stop Services

```bash
python manage_infrastructure.py --stop
```

Stops all services gracefully. Preserves data volumes.

### Restart Services

```bash
python manage_infrastructure.py --restart
```

Restarts all running services.

## Connection Information

After starting services, the management script displays connection details:

| Service | Connection String | Username | Password |
|---------|------------------|----------|----------|
| Milvus | localhost:19530 | - | - |
| Attu UI | http://localhost:8000 | - | - |
| MongoDB | mongodb://localhost:27017 | admin | password123 |
| MinIO | http://localhost:9000 | minioadmin | minioadmin |
| Redis | redis://localhost:6379 | - | redis123 |

## Architecture

### Network

All services run on the `asclepius-network` Docker network for internal communication. Services communicate using container names as hostnames.

### Data Persistence

Data is persisted in `./volumes/` directory:
- `volumes/etcd/`: etcd metadata
- `volumes/minio/`: MinIO object storage
- `volumes/milvus/`: Milvus data files
- `volumes/mongodb/`: MongoDB data files
- `volumes/redis/`: Redis persistence data

### Dependencies

- Milvus depends on etcd and MinIO
- Attu depends on Milvus standalone
- MongoDB and Redis operate independently

### Health Checks

All services include health check configurations. Docker Compose monitors service health and reports status.

## Project Identification

All services and networks are labeled with:
- `project=asclepius-healthcare-chatbot`
- Service-specific labels for filtering

Filter Docker resources:
```bash
docker ps --filter "label=project=asclepius-healthcare-chatbot"
docker network ls --filter "label=project=asclepius-healthcare-chatbot"
```

## Troubleshooting

### Port Conflicts

If ports are already in use, stop conflicting services or modify port mappings in `docker-compose.yml`.

### Container Conflicts

The management script automatically detects and handles containers from other projects with the same names. Conflicting containers are stopped and removed while preserving their volumes.

### Docker Not Running

Ensure Docker Desktop is running before executing management commands. The script validates Docker availability before operations.

### Volume Permissions

On Linux, ensure the Docker user has write permissions to the `./volumes/` directory.

## Configuration

### Environment Variables

- `DOCKER_VOLUME_DIRECTORY`: Override default volume path (default: `./volumes/`)

### Service Versions

- Milvus: 2.2.16
- MongoDB: 7.0
- Redis: 7.2-alpine
- etcd: 3.5.5
- MinIO: RELEASE.2023-03-20T20-16-18Z
- Attu: 2.2.8

## Files

- `docker-compose.yml`: Service definitions and configuration
- `manage_infrastructure.py`: Python script for service management
- `flush_redis.py`: Redis cache flush utility
- `volumes/`: Persistent data storage (git-ignored)

## Redis Cache Management

### Flush Utility

The `flush_redis.py` utility provides a production-grade tool for safely managing Redis cache data. It offers multiple flush strategies, comprehensive safety features, and detailed operation reporting.

#### Features

**Multiple Flush Strategies:**
- **Full Database Flush**: Delete all keys in the selected Redis database (use with caution)
- **Prefix Pattern Deletion**: Delete only keys matching a specific prefix pattern (default: `drug_reco*`)

**Safety Features:**
- **Dry Run Mode**: Preview operations without making changes (`--dry-run`)
- **Confirmation Prompts**: Interactive confirmation before deletion (can be skipped with `--no-confirm`)
- **Operation Preview**: Shows key count and sample keys before execution
- **Batch Processing**: Handles large key sets efficiently without memory issues
- **Connection Retry**: Automatic retry logic for transient connection failures

**Professional Output:**
- Color-coded terminal output (success/error/warning/info)
- Progress tracking for long-running operations
- Detailed statistics: keys deleted, duration, throughput, batches processed
- Structured headers and formatted information display

**Configuration Management:**
- Loads configuration from multiple sources with priority order
- Supports command-line arguments, environment variables, `.env` file, and defaults
- Automatic `.env` file detection in project root

#### Usage

**Basic Operations:**

```bash
# Preview what will be deleted (safe, no changes made)
python infrastructure/flush_redis.py --dry-run

# Delete keys with default prefix "drug_reco*"
python infrastructure/flush_redis.py

# Delete keys with custom prefix
python infrastructure/flush_redis.py --prefix "my_custom_prefix"

# Flush entire Redis database (WARNING: Deletes ALL keys!)
python infrastructure/flush_redis.py --full
```

**Advanced Options:**

```bash
# Skip confirmation prompt (use with caution)
python infrastructure/flush_redis.py --no-confirm

# Custom Redis connection
python infrastructure/flush_redis.py --host redis.example.com --port 6380 --db 1

# Specify custom password
python infrastructure/flush_redis.py --password mypassword

# Combine options: dry run with custom prefix
python infrastructure/flush_redis.py --dry-run --prefix "session_data"
```

**Command-Line Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--full` | Flush entire database (deletes ALL keys) | False |
| `--prefix PREFIX` | Key prefix for pattern matching | `drug_reco` |
| `--dry-run` | Preview operation without deleting | False |
| `--no-confirm` | Skip confirmation prompt | False |
| `--host HOST` | Redis server host | `localhost` |
| `--port PORT` | Redis server port | `6379` |
| `--db DB` | Redis database index | `0` |
| `--password PASSWORD` | Redis authentication password | From env/default |
| `--no-color` | Disable colored output | False |
| `--verbose` | Enable verbose output | False |

#### Configuration Priority

Configuration is loaded in the following priority order (highest to lowest):

1. **Command-line arguments** (highest priority)
2. **Environment variables** (`REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_KEY_PREFIX`)
3. **`.env` file** in project root
4. **Default values** (lowest priority)

**Environment Variables:**

```bash
REDIS_HOST=localhost          # Redis server host
REDIS_PORT=6379              # Redis server port
REDIS_DB=0                   # Database index
REDIS_PASSWORD=redis123      # Authentication password
REDIS_KEY_PREFIX=drug_reco   # Default key prefix
```

#### Operation Flow

1. **Load Configuration**: Reads settings from CLI args → Environment → `.env` → Defaults
2. **Connect to Redis**: Establishes connection with retry logic (3 attempts)
3. **Display Connection Info**: Shows Redis version, memory usage, total keys
4. **Preview Operation**: Scans and counts keys that will be affected, shows samples
5. **Confirmation**: Prompts user for confirmation (unless `--no-confirm` or `--dry-run`)
6. **Execute Flush**: Performs deletion with progress updates
7. **Display Results**: Shows comprehensive statistics and operation summary

#### Operation Statistics

After execution, the utility displays detailed statistics:

- **Strategy**: Operation type (full_database or prefix_pattern)
- **Pattern**: Key pattern used (if applicable)
- **Keys Before**: Total keys before operation
- **Keys Deleted**: Number of keys removed
- **Keys After**: Remaining keys after operation
- **Duration**: Operation time in seconds
- **Throughput**: Keys deleted per second
- **Batches Processed**: Number of batch operations

#### Examples

**Preview deletion before committing:**
```bash
python infrastructure/flush_redis.py --dry-run
```
Shows what would be deleted without making changes.

**Delete all application cache keys:**
```bash
python infrastructure/flush_redis.py
```
Deletes all keys matching `drug_reco*` pattern with confirmation prompt.

**Quick cleanup without confirmation:**
```bash
python infrastructure/flush_redis.py --no-confirm
```
Deletes default prefix keys immediately (use with caution).

**Custom prefix cleanup:**
```bash
python infrastructure/flush_redis.py --prefix "temp_data"
```
Deletes all keys starting with `temp_data*`.

**Full database reset:**
```bash
python infrastructure/flush_redis.py --full
```
**Warning**: This deletes ALL keys in the selected database. Use only when you need to completely reset the cache.

**Remote Redis instance:**
```bash
python infrastructure/flush_redis.py --host 192.168.1.100 --port 6380 --db 1 --password mypass
```
Connects to a remote Redis instance with custom credentials.

#### Error Handling

The utility provides clear error messages for common issues:

- **Connection failures**: Automatic retry with helpful suggestions
- **Authentication errors**: Clear password validation messages
- **Invalid arguments**: Validation with helpful error messages
- **Missing dependencies**: Instructions for installing required packages

#### Requirements

- Python 3.9+
- `redis` package: `pip install redis`
- `python-dotenv` package (optional, for `.env` file support): `pip install python-dotenv`




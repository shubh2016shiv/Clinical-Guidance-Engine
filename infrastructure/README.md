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
- `volumes/`: Persistent data storage (git-ignored)


# Cache Provider Module

Comprehensive caching solution for the Drug Recommendation Chatbot, providing a clean, maintainable interface for all Redis caching operations.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Operation Guides](#operation-guides)
- [TTL Strategy](#ttl-strategy)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Integration Examples](#integration-examples)
- [API Reference](#api-reference)

## Overview

The Cache Provider module implements a provider pattern for caching operations, currently supporting Redis with plans for additional backends (Memcached, KeyDB). It provides:

- **Session Management**: Store and retrieve session metadata with automatic TTL management
- **Response Chain Tracking**: Maintain conversation history chains with automatic trimming
- **History Caching**: Cache conversation history to reduce database queries
- **Tool Execution State**: Track in-progress tool calls for recovery and monitoring
- **Streaming State**: Manage temporary state for active streaming sessions
- **Vector Store Validation**: Cache validation results to avoid repeated API calls
- **Rate Limiting**: Sliding window rate limiting with sorted sets
- **Request Correlation**: Link requests to responses for debugging

## Architecture

### Provider Pattern

The module follows the same provider pattern used by LLM and embedding providers:

```
CacheProvider (Abstract)
    ├── RedisProvider (Implementation)
    ├── MemcachedProvider (Future)
    └── KeyDBProvider (Future)
```

### Key Naming Convention

All cache keys follow a consistent naming pattern:
```
{key_prefix}:{operation_type}:{identifier1}:{identifier2}:...
```

Examples:
- `drug_reco:session:uuid-123`
- `drug_reco:response_chain:uuid-123`
- `drug_reco:history:uuid-123`
- `drug_reco:tool_execution:uuid-123:req-456`

### Data Structures

Redis data structures are chosen based on access patterns:

| Operation | Redis Structure | Reason |
|-----------|----------------|---------|
| Session Metadata | Hash | Multiple fields, partial updates |
| Response Chain | List | Ordered, append operations, auto-trim |
| History Cache | String (JSON) | Full replacement on update |
| Tool Execution State | Hash | Multiple fields, status tracking |
| Rate Limiting | Sorted Set | Timestamp-based sliding window |
| Simple Mappings | String | Single value lookups |

## Installation

### Dependencies

Add to `requirements.txt`:
```
redis>=5.0.0
```

For better performance (optional):
```
redis[hiredis]>=5.0.0
```

### Redis Server

The chatbot uses Docker Compose for Redis:

```bash
# Start Redis
cd infrastructure
docker-compose up -d redis

# Verify Redis is running
docker-compose ps redis
```

Redis configuration (from `docker-compose.yml`):
- Host: `localhost`
- Port: `6379`
- Password: `redis123`
- Version: `7.2-alpine`

## Configuration

### Environment Variables

Add to `.env` file:

```env
# Redis Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=redis123

# Connection Pool
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5

# Key Management
REDIS_KEY_PREFIX=drug_reco

# Default TTLs (in seconds)
REDIS_DEFAULT_SESSION_TTL=7200    # 2 hours
REDIS_DEFAULT_HISTORY_TTL=1800    # 30 minutes
```

### Configuration Object

```python
from src.providers.cache_provider import CacheConfig

config = CacheConfig(
    host="localhost",
    port=6379,
    db=0,
    password="redis123",
    max_connections=50,
    socket_timeout=5,
    key_prefix="drug_reco",
    default_session_ttl=7200,
    default_history_ttl=1800,
    default_tool_execution_ttl=300,
    default_streaming_ttl=120,
    default_vector_store_ttl=86400,
    default_rate_limit_ttl=3600,
    default_request_correlation_ttl=3600,
    max_chain_length=20,
)
```

## Quick Start

### Basic Usage

```python
from src.providers.cache_provider import create_cache_provider

# Create provider (reads from config/environment)
cache_provider = create_cache_provider()

# Connect to Redis
await cache_provider.connect()

# Store session metadata
await cache_provider.set_session_metadata(
    session_id="session-123",
    metadata={
        "user_id": "user-456",
        "vector_store_id": "vs-789",
        "created_at": "2024-01-01T00:00:00Z",
        "last_active_at": "2024-01-01T12:00:00Z",
        "message_count": 5
    }
)

# Retrieve session metadata
metadata = await cache_provider.get_session_metadata("session-123")
print(f"Session metadata: {metadata}")

# Append to response chain
await cache_provider.append_response("session-123", "resp-001")
await cache_provider.append_response("session-123", "resp-002")

# Get response chain
chain = await cache_provider.get_response_chain("session-123")
print(f"Response chain: {chain}")  # ['resp-001', 'resp-002']

# Disconnect when done
await cache_provider.disconnect()
```

### With Custom Configuration

```python
from src.providers.cache_provider import create_cache_provider

# Create with custom settings
cache_provider = create_cache_provider(
    provider_type="redis",
    host="redis-server.example.com",
    port=6379,
    password="secure_password",
    key_prefix="my_app",
    default_session_ttl=3600  # 1 hour
)

await cache_provider.connect()
# ... use provider
await cache_provider.disconnect()
```

## Operation Guides

### Session Management

```python
# Store session metadata
await cache_provider.set_session_metadata(
    session_id="session-123",
    metadata={
        "user_id": "user-456",
        "vector_store_id": "vs-789",
        "agent_ready": True,
        "created_at": "2024-01-01T00:00:00Z"
    },
    ttl=7200  # Optional, uses default if not specified
)

# Retrieve session metadata
metadata = await cache_provider.get_session_metadata("session-123")

# Extend session TTL (keep session alive)
await cache_provider.touch_session("session-123")

# Delete all session data
await cache_provider.delete_session("session-123")
```

### Response Chain Tracking

```python
# Append response to chain
await cache_provider.append_response("session-123", "resp-001")

# Chain is automatically trimmed to max_chain_length (default: 20)
for i in range(25):
    await cache_provider.append_response("session-123", f"resp-{i:03d}")

# Get complete chain (only last 20 entries)
chain = await cache_provider.get_response_chain("session-123")

# Get last response quickly (O(1) operation)
last_response_id = await cache_provider.get_last_response_id("session-123")

# Set last response ID
await cache_provider.set_last_response_id("session-123", "resp-025")
```

### History Caching

```python
# Cache conversation history
history = [
    {"role": "user", "content": "What is metformin?"},
    {"role": "assistant", "content": "Metformin is..."}
]

await cache_provider.cache_history(
    session_id="session-123",
    history=history,
    ttl=1800  # 30 minutes
)

# Retrieve cached history
cached_history = await cache_provider.get_cached_history("session-123")

if cached_history:
    print("Cache hit!")
    # Use cached history
else:
    print("Cache miss, query database")
    # Query database and cache result

# Invalidate cache when history changes
await cache_provider.invalidate_history("session-123")
```

### Tool Execution State

```python
# Track tool execution
await cache_provider.set_tool_execution_state(
    session_id="session-123",
    request_id="req-456",
    state={
        "status": "in_progress",
        "tool_calls": [
            {
                "call_id": "call_abc123",
                "function_name": "search_drug_database",
                "status": "executing",
                "started_at": "2024-01-01T12:00:00Z"
            }
        ],
        "started_at": "2024-01-01T12:00:00Z"
    },
    ttl=300  # 5 minutes
)

# Check tool execution state
state = await cache_provider.get_tool_execution_state("session-123", "req-456")

if state and state["status"] == "in_progress":
    print("Tool execution in progress...")

# Clear state after completion
await cache_provider.clear_tool_execution_state("session-123", "req-456")
```

### Streaming State

```python
# Track active streaming session
await cache_provider.set_streaming_state(
    session_id="session-123",
    request_id="req-456",
    state={
        "response_id": "resp-789",
        "status": "streaming",
        "chunk_count": 42,
        "last_chunk_at": "2024-01-01T12:00:05Z",
        "is_citation": False
    },
    ttl=120  # 2 minutes
)

# Check streaming state
state = await cache_provider.get_streaming_state("session-123", "req-456")

# Clear after stream completion
await cache_provider.clear_streaming_state("session-123", "req-456")
```

### Vector Store Validation

```python
# Cache validation results (long TTL)
await cache_provider.cache_vector_store_validation(
    vector_store_id="vs-789",
    validation_data={
        "status": "completed",
        "validated_at": "2024-01-01T10:00:00Z",
        "file_count": 15,
        "expires_at": "2024-01-02T10:00:00Z"
    },
    ttl=86400  # 24 hours
)

# Check cached validation
validation = await cache_provider.get_vector_store_validation("vs-789")

if validation:
    print(f"Vector store status: {validation['status']}")
else:
    # Validate with OpenAI API and cache result
    pass
```

### Rate Limiting

```python
# Increment request count (sliding window)
count = await cache_provider.increment_request_count(
    session_id="session-123",
    window_seconds=3600  # 1 hour window
)

print(f"Requests in last hour: {count}")

# Check rate limit
within_limit = await cache_provider.check_rate_limit(
    session_id="session-123",
    limit=100,
    window_seconds=3600
)

if not within_limit:
    raise Exception("Rate limit exceeded")

# Get current count without incrementing
count = await cache_provider.get_request_count("session-123", window_seconds=3600)
```

### Request Correlation

```python
# Store request correlation for debugging
await cache_provider.set_request_correlation(
    session_id="session-123",
    request_id="req-456",
    correlation_data={
        "response_id": "resp-789",
        "user_message": "What is metformin?",
        "status": "completed",
        "created_at": "2024-01-01T12:00:00Z"
    },
    ttl=3600  # 1 hour
)

# Retrieve correlation data
correlation = await cache_provider.get_request_correlation("session-123", "req-456")
```

### Chat ID Mapping

```python
# Replace in-memory _chat_cache in ChatManager
await cache_provider.set_chat_mapping(
    chat_id="chat-123",
    response_id="resp-789",
    ttl=7200
)

# Retrieve mapping
response_id = await cache_provider.get_chat_mapping("chat-123")

if response_id:
    # Use response_id for chaining
    pass
```

## TTL Strategy

### TTL Decision Matrix

| Data Type | Default TTL | Strategy | Rationale |
|-----------|-------------|----------|-----------|
| Session Metadata | 2 hours (7200s) | Extend on activity | Active sessions need persistence |
| Response Chain | 2 hours (7200s) | Same as session | Tied to session lifecycle |
| Last Response ID | 2 hours (7200s) | Same as session | Hot path data |
| Conversation History | 30 min (1800s) | Invalidate on update | Larger payload, frequent updates |
| Vector Store Validation | 24 hours (86400s) | Fixed | Stable, expensive to validate |
| Tool Execution State | 5 min (300s) | Auto-expire | Temporary state |
| Streaming State | 2 min (120s) | Auto-expire | Very temporary |
| Rate Limiting | 1 hour (3600s) | Rolling window | Security concern |
| Chat ID Mapping | 2 hours (7200s) | Same as session | Tied to session |
| Request Correlation | 1 hour (3600s) | Fixed | Debugging, shorter retention |

### Activity-Based Extension

```python
# Extend session TTL on each request
async def on_user_request(session_id: str):
    # Process request
    ...
    
    # Extend TTL for all session keys
    await cache_provider.touch_session(session_id)
```

### Invalidation Triggers

```python
# Invalidate history cache on new message
async def on_new_message(session_id: str, message: str):
    # Save message to database
    ...
    
    # Invalidate history cache
    await cache_provider.invalidate_history(session_id)
```

## Error Handling

### Exception Hierarchy

```python
CacheProviderError (base)
├── CacheConnectionError
│   └── CacheAuthenticationError
├── CacheConfigError
├── CacheOperationError
├── CacheSerializationError
├── CacheKeyError
└── CacheTTLError
```

### Handling Connection Errors

```python
from src.providers.cache_provider import (
    create_cache_provider,
    CacheConnectionError,
    CacheAuthenticationError
)

try:
    cache_provider = create_cache_provider()
    await cache_provider.connect()
except CacheAuthenticationError as e:
    logger.error(f"Redis authentication failed: {e}")
    # Check credentials
except CacheConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    # Fall back to in-memory cache or database
```

### Handling Operation Errors

```python
from src.providers.cache_provider import CacheOperationError

try:
    metadata = await cache_provider.get_session_metadata(session_id)
except CacheOperationError as e:
    logger.error(f"Cache operation failed: {e}")
    # Fall back to database query
    metadata = await db.get_session(session_id)
```

### Graceful Degradation

```python
async def get_session_metadata_with_fallback(session_id: str):
    """Get session metadata with cache fallback"""
    try:
        # Try cache first
        metadata = await cache_provider.get_session_metadata(session_id)
        if metadata:
            return metadata
    except Exception as e:
        logger.warning(f"Cache error, falling back to database: {e}")
    
    # Fall back to database
    return await db.get_session(session_id)
```

## Performance Considerations

### Connection Pooling

The Redis provider uses connection pooling for efficient connection management:

```python
config = CacheConfig(
    max_connections=50,  # Adjust based on concurrency
    socket_timeout=5,
    socket_connect_timeout=5
)
```

### Caching Provider Instances

```python
from src.providers.cache_provider import get_cached_cache_provider

# First call creates provider
provider1 = get_cached_cache_provider("redis", "localhost", 6379)

# Subsequent calls return cached instance
provider2 = get_cached_cache_provider("redis", "localhost", 6379)

# provider1 is provider2 == True
```

### Batch Operations

For multiple operations, use pipelining (future enhancement):

```python
# Future: Pipeline support
async with cache_provider.pipeline() as pipe:
    await pipe.set_session_metadata(session_id, metadata)
    await pipe.append_response(session_id, response_id)
    await pipe.execute()
```

### Memory Optimization

- Response chains automatically trim to `max_chain_length` (default: 20)
- Use appropriate TTLs to prevent memory bloat
- Monitor Redis memory usage: `redis-cli INFO memory`

## Integration Examples

### ChatManager Integration

Replace in-memory `_chat_cache` with Redis:

```python
from src.providers.cache_provider import create_cache_provider

class ChatManager:
    def __init__(self, cache_provider=None):
        self.cache_provider = cache_provider or create_cache_provider()
        # Remove: self._chat_cache: Dict[str, str] = {}
    
    async def create_chat(self, message: str, **kwargs):
        response = await self._create_response(message, **kwargs)
        chat_id = response.id
        
        # Store in Redis instead of in-memory dict
        await self.cache_provider.set_chat_mapping(chat_id, response.id)
        
        return chat_id
    
    async def continue_chat(self, chat_id: str, message: str, **kwargs):
        # Get from Redis instead of in-memory dict
        last_response_id = await self.cache_provider.get_chat_mapping(chat_id)
        
        if not last_response_id:
            last_response_id = chat_id
        
        response = await self._create_response(
            message,
            previous_response_id=last_response_id,
            **kwargs
        )
        
        # Update mapping
        await self.cache_provider.set_chat_mapping(chat_id, response.id)
        
        return response.id
```

### PersistenceManager Integration

Cache conversation history:

```python
from src.providers.cache_provider import create_cache_provider

class PersistenceManager:
    def __init__(self, cache_provider=None):
        self.cache_provider = cache_provider or create_cache_provider()
    
    async def load_conversation_history(self, session_id: str):
        # Try cache first
        history = await self.cache_provider.get_cached_history(session_id)
        
        if history:
            logger.debug(f"Cache hit for session {session_id}")
            return history
        
        # Cache miss - load from file/database
        logger.debug(f"Cache miss for session {session_id}")
        history = self._load_from_storage(session_id)
        
        # Cache for next time
        await self.cache_provider.cache_history(session_id, history)
        
        return history
    
    async def save_conversation(self, session_id: str, message: dict):
        # Save to persistent storage
        self._save_to_storage(session_id, message)
        
        # Invalidate cache
        await self.cache_provider.invalidate_history(session_id)
```

### StreamManager Integration

Track streaming state:

```python
from src.providers.cache_provider import create_cache_provider

class StreamManager:
    def __init__(self, cache_provider=None):
        self.cache_provider = cache_provider or create_cache_provider()
    
    async def stream_response(self, session_id: str, request_id: str, **kwargs):
        # Set initial streaming state
        await self.cache_provider.set_streaming_state(
            session_id=session_id,
            request_id=request_id,
            state={
                "status": "streaming",
                "chunk_count": 0,
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        try:
            chunk_count = 0
            async for chunk in self._stream_chunks(**kwargs):
                chunk_count += 1
                
                # Update state periodically
                if chunk_count % 10 == 0:
                    await self.cache_provider.set_streaming_state(
                        session_id=session_id,
                        request_id=request_id,
                        state={
                            "status": "streaming",
                            "chunk_count": chunk_count,
                            "last_chunk_at": datetime.utcnow().isoformat()
                        }
                    )
                
                yield chunk
        
        finally:
            # Clear state after completion
            await self.cache_provider.clear_streaming_state(session_id, request_id)
```

## API Reference

### Factory Functions

#### create_cache_provider()
```python
create_cache_provider(
    provider_type: Optional[str] = None,
    config: Optional[CacheConfig] = None,
    **kwargs
) -> CacheProvider
```

Main factory function for creating cache provider instances.

#### get_default_cache_provider()
```python
get_default_cache_provider() -> CacheProvider
```

Get provider with all defaults from configuration.

#### get_cached_cache_provider()
```python
get_cached_cache_provider(
    provider_type: str = "redis",
    host: str = "localhost",
    port: int = 6379,
    db: int = 0
) -> CacheProvider
```

Get or create a cached provider instance.

### Core Methods

See module docstrings for complete API documentation:
- Session Management: `get/set/touch/delete_session_metadata`
- Response Chain: `append_response`, `get_response_chain`, `get/set_last_response_id`
- History: `cache_history`, `get_cached_history`, `invalidate_history`
- Tool Execution: `set/get/clear_tool_execution_state`
- Streaming: `set/get/clear_streaming_state`
- Vector Store: `cache/get_vector_store_validation`
- Rate Limiting: `increment/get_request_count`, `check_rate_limit`
- Request Correlation: `set/get_request_correlation`
- Chat Mapping: `set/get_chat_mapping`
- Utilities: `exists`, `delete`, `expire`, `get_ttl`

## Future Enhancements

- [ ] Memcached provider implementation
- [ ] KeyDB provider implementation
- [ ] Redis Cluster support
- [ ] Pipeline/transaction support
- [ ] Cache statistics and metrics
- [ ] Cache warming strategies
- [ ] Compression for large values
- [ ] Multi-region replication

## Support

For issues or questions:
1. Check this README and module docstrings
2. Review exception messages and logs
3. Verify Redis connection and configuration
4. Check Docker Compose logs: `docker-compose logs redis`

## License

Part of the Drug Recommendation Chatbot project.


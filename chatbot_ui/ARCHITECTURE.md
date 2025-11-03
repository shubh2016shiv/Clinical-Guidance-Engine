# Chainlit UI Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Browser                                 │
│                     (http://localhost:8000)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               │ HTTP/WebSocket
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      Chainlit Framework                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    chatbot_ui/app.py                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │  │
│  │  │@cl.on_chat_ │  │@cl.on_      │  │@cl.on_chat_ │          │  │
│  │  │start        │  │message      │  │end          │          │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │  │
│  └─────────┼─────────────────┼─────────────────┼─────────────────┘  │
└────────────┼─────────────────┼─────────────────┼────────────────────┘
             │                 │                 │
             │                 │                 │
┌────────────▼─────────────────▼─────────────────▼────────────────────┐
│                   chatbot_ui/chat_handler.py                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  ChainlitChatHandler                                           │ │
│  │  ├─ handle_message() → AsyncGenerator[chunk]                  │ │
│  │  ├─ handle_message_non_streaming()                            │ │
│  │  └─ format_response_with_citations()                          │ │
│  └──────────────┬────────────────────────┬────────────────────────┘ │
└─────────────────┼────────────────────────┼──────────────────────────┘
                  │                        │
                  │                        │
        ┌─────────▼─────────┐    ┌────────▼─────────┐
        │  Persistence      │    │  Citation        │
        │  Manager          │    │  Formatter       │
        │                   │    │                  │
        │  - save_         │    │  - format_inline │
        │    conversation() │    │  - create_       │
        │  - load_         │    │    elements()    │
        │    conversation() │    │  - format_       │
        │                   │    │    section()     │
        └─────────┬─────────┘    └──────────────────┘
                  │
                  │ JSON Files
                  │
        ┌─────────▼─────────┐
        │  conversations/   │
        │  ├─ session1.json │
        │  ├─ session2.json │
        │  └─ session3.json │
        └───────────────────┘
                  
┌──────────────────────────────────────────────────────────────────────┐
│                        Response API Agent                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  src/response_api_agent/asclepius_agent.py                    │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │  AsclepiusHealthcareAgent                                │ │ │
│  │  │  ├─ initialize_knowledge_base()                          │ │ │
│  │  │  ├─ consult(streaming=True)                              │ │ │
│  │  │  └─ close_session()                                      │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  └────────────────────┬───────────────────┬──────────────────────┘ │
└───────────────────────┼───────────────────┼────────────────────────┘
                        │                   │
            ┌───────────▼─────────┐  ┌──────▼──────────┐
            │  OpenAI Responses   │  │  Drug Database  │
            │  API                │  │  (Milvus)       │
            │  - GPT-4o          │  │  - Vector Store │
            │  - Streaming       │  │  - Drug Data    │
            │  - Citations       │  │  - Semantic     │
            │                    │  │    Search       │
            └────────────────────┘  └─────────────────┘
                        │
            ┌───────────▼─────────┐
            │  Clinical Guidelines│
            │  Vector Store       │
            │  - Guidelines DB    │
            │  - File Citations   │
            └─────────────────────┘
```

## Data Flow Architecture

### Request Flow (User Message → Response)

```
1. User Input
   │
   ├─→ User types message in Chainlit UI
   │
   └─→ @cl.on_message triggered
       │
       └─→ chat_handler.handle_message()
           │
           ├─→ Generate request_id (UUID)
           │
           └─→ agent.consult(streaming=True)
               │
               ├─→ OpenAI Responses API (streaming)
               │   │
               │   ├─→ Clinical Guidelines Search (optional)
               │   ├─→ Drug Database Search (optional)
               │   └─→ Stream Response Chunks
               │
               └─→ Return AsyncGenerator
                   │
                   └─→ For each chunk:
                       ├─→ Extract text
                       ├─→ Extract response_id
                       ├─→ Stream to UI (cl.Message.stream_token)
                       └─→ Collect citations
                           │
                           └─→ On completion:
                               ├─→ Format citations
                               ├─→ Create side elements
                               ├─→ Save to persistence
                               └─→ Display complete message
```

### Response Streaming Flow

```
OpenAI API Stream
    │
    ├─→ [ResponseCreatedEvent]
    │   └─→ Capture response_id
    │
    ├─→ [ResponseTextDeltaEvent] (multiple)
    │   └─→ Stream chunks → UI
    │
    ├─→ [ResponseOutputTextAnnotationAddedEvent] (optional)
    │   └─→ Collect citation metadata
    │
    ├─→ [ResponseFileSearchCallCompleted] (optional)
    │   └─→ File search completed
    │
    └─→ [ResponseCompletedEvent]
        └─→ Retrieve complete response
            ├─→ Extract citations
            ├─→ Format for display
            └─→ Save conversation
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chainlit UI Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  app.py                                                          │
│  ├─ Session Management                                          │
│  ├─ Event Handlers                                              │
│  └─ Error Handling                                              │
│                                                                  │
│  config.py                  starter_prompts.py                  │
│  ├─ UI Settings             ├─ Welcome Starters                │
│  ├─ Display Options         └─ Example Questions               │
│  └─ Error Messages                                              │
│                                                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ delegates to
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    Integration Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  chat_handler.py                                                │
│  ├─ Message Processing                                          │
│  ├─ Stream Coordination                                         │
│  ├─ Citation Collection                                         │
│  └─ Persistence Coordination                                    │
│                                                                  │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ citation_formatter  │        │ persistence_manager │        │
│  │                     │        │                     │        │
│  │ - Inline citations  │        │ - JSON storage     │        │
│  │ - Side elements     │        │ - History tracking │        │
│  │ - References        │        │ - Cleanup          │        │
│  └─────────────────────┘        └─────────────────────┘        │
│                                                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ calls
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    Agent Layer (Existing)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  src/response_api_agent/asclepius_agent.py                     │
│  ├─ Healthcare Query Processing                                 │
│  ├─ Knowledge Base Management                                   │
│  └─ Multi-turn Conversation                                     │
│                                                                  │
│  src/response_api_agent/managers/                              │
│  ├─ response_api_manager.py     (API coordination)            │
│  ├─ stream_manager.py           (streaming handling)          │
│  ├─ chat_manager.py             (conversation management)      │
│  ├─ tool_manager.py             (tool execution)               │
│  ├─ drug_data_manager.py        (Milvus integration)          │
│  └─ citation_manager.py         (citation extraction)          │
│                                                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ uses
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                  External Services                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OpenAI Responses API          Milvus Vector DB                 │
│  ├─ GPT-4o Model              ├─ Drug Database                 │
│  ├─ Streaming Support         ├─ Clinical Guidelines           │
│  └─ Citation Generation       └─ Semantic Search               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Session Lifecycle

```
User Connects
    │
    ├─→ @cl.on_chat_start
    │   │
    │   ├─→ Generate session_id (UUID)
    │   ├─→ Initialize AsclepiusHealthcareAgent
    │   ├─→ Setup knowledge base (async)
    │   │   ├─→ Load clinical guidelines
    │   │   └─→ Create vector store
    │   ├─→ Store agent in cl.user_session
    │   └─→ Send welcome message + starters
    │
    ├─→ User Interaction Loop
    │   │
    │   ├─→ User sends message
    │   │   │
    │   │   ├─→ @cl.on_message
    │   │   ├─→ Process with agent (streaming)
    │   │   ├─→ Display response chunks
    │   │   ├─→ Show citations
    │   │   └─→ Save to persistence
    │   │
    │   └─→ Repeat for each message
    │
    └─→ User Disconnects
        │
        ├─→ @cl.on_chat_end
        │   │
        │   ├─→ Retrieve agent from session
        │   ├─→ Call agent.close_session()
        │   │   ├─→ Cleanup vector stores (if enabled)
        │   │   └─→ Clear conversation cache
        │   └─→ Log session end
        │
        └─→ Session cleanup complete
```

## Persistence Architecture

```
Conversation Storage (JSON-based)

┌──────────────────────────────────────────────┐
│  chatbot_ui/conversations/                   │
│                                              │
│  ├─ {session_id_1}.json                     │
│  │  {                                        │
│  │    "session_id": "uuid",                 │
│  │    "created_at": "timestamp",            │
│  │    "updated_at": "timestamp",            │
│  │    "messages": [                         │
│  │      {                                    │
│  │        "request_id": "msg_user_123",     │
│  │        "response_id": null,              │
│  │        "role": "user",                   │
│  │        "content": "What is metformin?",  │
│  │        "timestamp": "timestamp",         │
│  │        "metadata": {}                    │
│  │      },                                   │
│  │      {                                    │
│  │        "request_id": "msg_user_123",     │ ← Correlation
│  │        "response_id": "resp_xyz789",     │
│  │        "role": "assistant",              │
│  │        "content": "Metformin is...",     │
│  │        "timestamp": "timestamp",         │
│  │        "metadata": {                     │
│  │          "citations": [...],             │
│  │          "guidelines_used": true,        │
│  │          "drug_database_used": true     │
│  │        }                                  │
│  │      }                                    │
│  │    ]                                      │
│  │  }                                        │
│  │                                           │
│  ├─ {session_id_2}.json                     │
│  └─ {session_id_3}.json                     │
│                                              │
└──────────────────────────────────────────────┘

Correlation IDs:
- session_id: Unique per chat session
- request_id: Links user message to assistant response
- response_id: OpenAI Response API response ID
```

## Citation Flow

```
Response with Citations
    │
    ├─→ Agent consult completes
    │   │
    │   ├─→ Citations in result metadata
    │   └─→ Response has citation markers [1], [2]
    │
    ├─→ citation_formatter.format_combined_citations()
    │   │
    │   ├─→ Inline Processing
    │   │   ├─→ Preserve [1], [2] markers in text
    │   │   └─→ Keep text readable
    │   │
    │   ├─→ Side Elements
    │   │   ├─→ Create cl.Text elements
    │   │   ├─→ Add filename, quote
    │   │   └─→ Display="side"
    │   │
    │   └─→ References Section
    │       ├─→ Format as markdown
    │       ├─→ Number citations [1], [2]
    │       └─→ Append to response
    │
    └─→ Display in UI
        ├─→ Main message with inline markers
        ├─→ Side panel with expandable citations
        └─→ References at bottom
```

## Error Handling Architecture

```
Error Occurs
    │
    ├─→ Agent Layer Error
    │   ├─→ Catch ResponsesAPIError
    │   ├─→ Log with structured logging
    │   └─→ Return user-friendly message
    │
    ├─→ UI Layer Error
    │   ├─→ Catch in @cl.on_message
    │   ├─→ Display error message
    │   └─→ Maintain session state
    │
    ├─→ Persistence Error
    │   ├─→ Catch IOError/JSONError
    │   ├─→ Log error
    │   └─→ Continue (non-critical)
    │
    └─→ Streaming Error
        ├─→ Catch during chunk processing
        ├─→ Yield error chunk
        └─→ Graceful completion
```

## Configuration Hierarchy

```
┌─────────────────────────────────────────────┐
│  Environment Variables (.env)                │
│  - OPENAI_API_KEY                           │
│  - GEMINI_API_KEY                           │
│  - MILVUS_HOST/PORT                         │
│  - ENABLE_STREAMING                         │
│  - ENABLE_CLEANUP                           │
└──────────────┬──────────────────────────────┘
               │ loaded by
               │
┌──────────────▼──────────────────────────────┐
│  src/config.py (Global Settings)            │
│  - get_settings() → Settings                │
│  - Agent configuration                      │
│  - API configuration                        │
└──────────────┬──────────────────────────────┘
               │ used by
               │
┌──────────────▼──────────────────────────────┐
│  chatbot_ui/config.py (UI Settings)         │
│  - APP_TITLE                                │
│  - SHOW_CITATION_ELEMENTS                   │
│  - STREAM_RESPONSES                         │
│  - MAX_HISTORY_MESSAGES                     │
└──────────────┬──────────────────────────────┘
               │ used by
               │
┌──────────────▼──────────────────────────────┐
│  chatbot_ui/.chainlit (Chainlit Config)    │
│  - UI theme                                 │
│  - Session timeout                          │
│  - Feature flags                            │
└─────────────────────────────────────────────┘
```

## Scalability Considerations

### Current Architecture (JSON-based)

```
Single Instance Deployment
    │
    ├─→ One Chainlit server
    ├─→ JSON file storage (local disk)
    ├─→ In-memory session management
    └─→ Limited to single server capacity

Limitations:
- No horizontal scaling
- File I/O for each conversation save
- No distributed session management
```

### Future Architecture (MongoDB-based)

```
Multi-Instance Deployment
    │
    ├─→ Multiple Chainlit servers
    │   ├─→ Load balancer
    │   └─→ Sticky sessions (optional)
    │
    ├─→ MongoDB cluster
    │   ├─→ Shared conversation storage
    │   ├─→ Horizontal scaling
    │   └─→ Replication
    │
    └─→ Session store (Redis)
        ├─→ Distributed sessions
        └─→ Cross-server compatibility

Benefits:
- Horizontal scaling
- High availability
- Better performance
- Centralized storage
```

## Security Architecture

### Current (Portfolio Project)

```
User → Chainlit UI → Agent → OpenAI API
       (No Auth)     (API Key in .env)
```

### Recommended (Production)

```
User → [Auth Layer] → Chainlit UI → Agent → OpenAI API
       (JWT/OAuth)    (Rate Limit)   (Secrets Mgmt)
                      (Input Valid)
```

## Technology Stack

```
Frontend:
  ├─ Chainlit Framework (v2.0+)
  ├─ WebSocket (streaming)
  └─ Modern browser (HTML5)

Backend:
  ├─ Python 3.9+
  ├─ AsyncIO (async/await)
  ├─ Pydantic (validation)
  └─ StructLog (logging)

Agent:
  ├─ OpenAI Responses API (GPT-4o)
  ├─ Custom streaming implementation
  └─ Tool execution framework

Storage:
  ├─ JSON files (conversations)
  ├─ Milvus (drug database)
  └─ Vector Store (clinical guidelines)

External Services:
  ├─ OpenAI API (LLM)
  ├─ Google Gemini (embeddings)
  └─ Milvus (vector search)
```

## Performance Optimization Points

```
1. Streaming
   ├─ Token-by-token display
   ├─ Async generators
   └─ Minimal latency

2. Citations
   ├─ Post-stream extraction
   ├─ Cached formatting
   └─ Lazy loading

3. Persistence
   ├─ Async file writes
   ├─ Message pruning
   └─ Batch updates (future)

4. Agent
   ├─ Connection pooling
   ├─ Vector store caching
   └─ Response caching (optional)
```

---

This architecture provides a solid foundation for the Chainlit UI integration while maintaining flexibility for future enhancements and scalability improvements.


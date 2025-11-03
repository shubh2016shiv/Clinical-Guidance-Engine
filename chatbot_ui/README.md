# Chainlit UI for Drug Recommendation Chatbot

Professional web-based user interface for the Asclepius Healthcare AI Assistant, built with the Chainlit framework.

## Overview

This module provides a clean, intuitive chat interface that integrates seamlessly with the AsclepiusHealthcareAgent, offering:

- Real-time streaming responses for immediate user feedback
- Citation display from clinical guidelines and drug databases
- Conversation persistence with correlation IDs (JSON-based)
- Starter prompts to guide users with common medical questions
- Professional error handling and logging

## Architecture

### Component Structure

```
chatbot_ui/
├── app.py                    # Main Chainlit application with lifecycle handlers
├── chat_handler.py           # Message processing and agent integration
├── persistence_manager.py    # JSON-based conversation storage
├── citation_formatter.py     # Citation formatting for display
├── starter_prompts.py        # Predefined medical question templates
├── config.py                 # UI-specific configuration
├── conversations/            # JSON conversation storage directory
└── README.md                # This file
```

### Design Principles

1. **Separation of Concerns**: UI logic isolated from agent implementation
2. **Clean Integration**: Thin adapter layer between Chainlit and AsclepiusHealthcareAgent
3. **Robust Error Handling**: Graceful failures with user-friendly messages
4. **Scalable Architecture**: Easy migration path from JSON to MongoDB persistence
5. **Professional UX**: Real-time streaming, clear citations, helpful starters

## Getting Started

### Prerequisites

Ensure you have completed the main project setup:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   ```

3. Ensure Milvus vector database is running and populated with drug data

### Running the Application

Start the Chainlit UI application:

```bash
chainlit run chatbot_ui/app.py
```

The application will be available at `http://localhost:8000` by default.

### Command Line Options

Chainlit supports various command-line options:

```bash
# Run on a specific port
chainlit run chatbot_ui/app.py --port 8080

# Run in watch mode (auto-reload on file changes)
chainlit run chatbot_ui/app.py --watch

# Run with custom host
chainlit run chatbot_ui/app.py --host 0.0.0.0
```

## Key Features

### 1. Real-Time Streaming

The UI streams responses token-by-token as the agent generates them, providing immediate feedback to users. This is powered by the agent's streaming capabilities integrated through the `ChainlitChatHandler`.

**Implementation**: `chatbot_ui/app.py` - `@cl.on_message` handler

### 2. Citation Display

Citations from clinical guidelines appear in two formats:

- **Inline**: Citation markers `[1]`, `[2]` in the response text
- **Side Elements**: Expandable citation cards with source filename and relevant excerpts
- **References Section**: Formatted list appended to response

**Implementation**: `chatbot_ui/citation_formatter.py` - `ChainlitCitationFormatter` class

### 3. Conversation Persistence

All conversations are persisted to JSON files with correlation IDs:

- **Request ID**: Unique identifier for each user message
- **Response ID**: Response ID from OpenAI Responses API
- **Session ID**: Unique identifier for each chat session

**Storage Format**:
```json
{
  "session_id": "uuid",
  "created_at": "2025-11-04T01:00:00.000000",
  "updated_at": "2025-11-04T01:05:00.000000",
  "messages": [
    {
      "request_id": "msg_user_abc123",
      "response_id": "resp_xyz789",
      "role": "user",
      "content": "What is metformin?",
      "timestamp": "2025-11-04T01:00:00.000000",
      "metadata": {}
    },
    {
      "request_id": "msg_user_abc123",
      "response_id": "resp_xyz789",
      "role": "assistant",
      "content": "Metformin is an oral antidiabetic medication...",
      "timestamp": "2025-11-04T01:00:05.000000",
      "metadata": {
        "citations": [...],
        "guidelines_used": true,
        "drug_database_used": true
      }
    }
  ]
}
```

**Implementation**: `chatbot_ui/persistence_manager.py` - `ConversationPersistenceManager` class

### 4. Starter Prompts

Users are greeted with example questions covering common medical topics:

- Drug formulations and dosages
- Drug interactions
- Clinical treatment guidelines
- Medication side effects
- Therapeutic recommendations

**Implementation**: `chatbot_ui/starter_prompts.py` - `get_starter_prompts()` function

## Configuration

### UI Settings

Edit `chatbot_ui/config.py` to customize the application:

```python
# Application Metadata
APP_TITLE = "Asclepius Healthcare AI Assistant"
APP_DESCRIPTION = "Evidence-based medical guidance..."

# Persistence Settings
CHAT_HISTORY_ENABLED = True
MAX_HISTORY_MESSAGES = 50
CONVERSATION_DIR = Path("chatbot_ui/conversations")

# Agent Configuration
AGENT_CHAT_HISTORY_LIMIT = 10
ENABLE_CLINICAL_GUIDELINES = True
ENABLE_DRUG_DATABASE = True

# Display Settings
SHOW_INLINE_CITATIONS = True
SHOW_CITATION_ELEMENTS = True
STREAM_RESPONSES = True
```

### Agent Settings

The agent uses settings from `src/config.py` loaded via `.env`:

```env
# Streaming configuration
ENABLE_STREAMING=true
ENABLE_CLEANUP=false

# Model configuration
OPENAI_MODEL_NAME=gpt-4o
OPENAI_TEMPERATURE=0.0
```

## Component Details

### Main Application (`app.py`)

The entry point for the Chainlit application with lifecycle event handlers:

**Event Handlers**:
- `@cl.on_chat_start`: Initialize agent and knowledge base
- `@cl.on_message`: Process user messages and stream responses
- `@cl.on_chat_end`: Clean up agent resources
- `@cl.on_stop`: Handle user-initiated stop during streaming
- `@cl.set_starters`: Configure starter prompts

**Key Responsibilities**:
- Agent initialization with knowledge base setup
- Session management with unique session IDs
- Error handling and user feedback
- Streaming coordination

### Chat Handler (`chat_handler.py`)

Bridges Chainlit UI with the AsclepiusHealthcareAgent:

**Key Methods**:
- `handle_message()`: Async generator for streaming responses
- `handle_message_non_streaming()`: Fallback for non-streaming responses
- `get_conversation_history()`: Retrieve session history
- `format_response_with_citations()`: Format citations for display

**Responsibilities**:
- Call agent's `consult()` method with appropriate parameters
- Process streaming chunks from agent
- Collect citations and metadata
- Save conversations to persistence layer

### Persistence Manager (`persistence_manager.py`)

Manages JSON-based conversation storage:

**Key Methods**:
- `save_conversation()`: Save user/assistant message pair
- `load_conversation()`: Load conversation from storage
- `load_conversation_history()`: Get message history
- `get_conversation_summary()`: Get conversation metadata
- `delete_conversation()`: Remove conversation
- `list_conversations()`: List all stored conversations
- `cleanup_old_conversations()`: Remove old conversations

**Storage Location**: `chatbot_ui/conversations/` directory

### Citation Formatter (`citation_formatter.py`)

Formats citations for Chainlit display:

**Key Methods**:
- `format_inline_citations()`: Replace markers with markdown links
- `create_citation_elements()`: Create Chainlit Text elements
- `format_citations_section()`: Create references section
- `format_combined_citations()`: Combine inline and side elements

**Citation Display Strategy**:
1. Inline markers remain in text
2. Side elements show expandable citation details
3. References section appended at end of response

### Starter Prompts (`starter_prompts.py`)

Provides predefined medical question templates:

**Key Functions**:
- `get_starter_prompts()`: Full list of starter prompts
- `get_welcome_starters()`: Curated subset for welcome screen

**Categories**:
- Drug information queries
- Drug interaction questions
- Clinical guideline questions
- Side effect inquiries
- Treatment recommendations

## Integration with Response API Agent

The UI integrates seamlessly with the existing agent infrastructure:

### Agent Initialization

```python
# From chatbot_ui/app.py
agent = AsclepiusHealthcareAgent(chat_history_limit=10)
vector_store_id = await agent.initialize_knowledge_base()
```

### Streaming Integration

```python
# From chatbot_ui/chat_handler.py
result = await agent.consult(
    query=message,
    use_clinical_guidelines=True,
    use_drug_database=True,
    streaming=True,
    enable_tool_execution=True
)

stream_generator = result["stream_generator"]

async for chunk_data in stream_generator:
    chunk_text = chunk_data["chunk"]
    response_id = chunk_data["conversation_id"]
    is_citation = chunk_data["is_citation"]
    # Process and display chunk
```

### Response Structure

Each streaming chunk contains:

```python
{
    "chunk": "text content",
    "conversation_id": "resp_abc123",  # Response API response ID
    "is_citation": false
}
```

## Development and Debugging

### Logging

The UI uses the structured logging system from `src/logs.py`:

```python
from src.logs import get_component_logger

logger = get_component_logger("ChainlitUI")
logger.info("Event occurred", component="ChainlitUI", subcomponent="OnMessage")
```

### Debug Mode

Enable debug mode in `.env`:

```env
DEBUG_MODE=true
```

### Common Issues

1. **Agent not initializing**:
   - Check OpenAI API key is set
   - Verify Milvus is running and accessible
   - Check clinical guidelines directory exists

2. **Streaming not working**:
   - Ensure `ENABLE_STREAMING=true` in `.env`
   - Check agent configuration in `src/config.py`
   - Review logs for streaming errors

3. **Citations not displaying**:
   - Verify clinical guidelines are loaded
   - Check `SHOW_CITATION_ELEMENTS=True` in config
   - Review citation extraction in logs

4. **Persistence not saving**:
   - Ensure `chatbot_ui/conversations/` directory exists
   - Check file permissions
   - Review error logs in console

### Testing the Integration

Test the UI with various scenarios:

1. **Basic Query**: "What is metformin?"
2. **Drug Interaction**: "Can you explain drug interactions for warfarin?"
3. **Guideline Query**: "What are the guidelines for treating type 2 diabetes?"
4. **Complex Query**: "What are the formulations and dosages for ACE inhibitors?"

## Migration Path to MongoDB

The current JSON-based persistence can be easily migrated to MongoDB:

1. Create MongoDB persistence manager implementing same interface
2. Replace `ConversationPersistenceManager` initialization in `chat_handler.py`
3. Update `chatbot_ui/config.py` with MongoDB connection settings
4. No changes needed to `app.py` or other components

**Planned MongoDB Schema**:
```javascript
{
  _id: ObjectId,
  session_id: String,
  created_at: ISODate,
  updated_at: ISODate,
  messages: [
    {
      request_id: String,
      response_id: String,
      role: String,
      content: String,
      timestamp: ISODate,
      metadata: Object
    }
  ]
}
```

## Performance Considerations

### Streaming Optimization

- Responses stream token-by-token for real-time feedback
- Citations retrieved after streaming completes
- Minimal latency between agent and UI

### Storage Optimization

- Conversations auto-pruned at `MAX_HISTORY_MESSAGES` limit
- Old conversations can be cleaned up with `cleanup_old_conversations()`
- JSON files use UTF-8 encoding for international character support

### Resource Management

- Agent resources cleaned up on session end (when `ENABLE_CLEANUP=true`)
- Vector stores persist across sessions for performance
- Conversation files written asynchronously

## Security Considerations

- No authentication required (portfolio project)
- Session IDs sanitized to prevent directory traversal
- API keys loaded from environment variables
- No sensitive data logged

## Future Enhancements

Potential improvements for production deployment:

1. **Authentication**: Add user authentication (password/OAuth)
2. **MongoDB Integration**: Replace JSON with MongoDB for scalability
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Conversation Search**: Add search functionality across conversations
5. **Export Conversations**: Allow users to download conversation history
6. **Multi-language Support**: Add internationalization
7. **Advanced Analytics**: Track usage patterns and popular queries
8. **Feedback Collection**: Allow users to rate responses

## Troubleshooting

### Application Won't Start

```bash
# Check Python version (requires 3.9+)
python --version

# Verify all dependencies installed
pip install -r requirements.txt

# Check if Milvus is running
# Visit http://localhost:9091 (Milvus Admin UI)
```

### Streaming Issues

```bash
# Check streaming configuration
cat .env | grep ENABLE_STREAMING

# Review application logs
# Logs appear in console when running chainlit
```

### Conversation Not Persisting

```bash
# Check conversations directory exists
ls -la chatbot_ui/conversations/

# Create directory if missing
mkdir -p chatbot_ui/conversations

# Check permissions
chmod 755 chatbot_ui/conversations
```

## Support

For issues or questions:

1. Check the logs in console output
2. Review the main project README for agent configuration
3. Verify all environment variables are set correctly
4. Check Milvus connection and data availability

## License

This component follows the same license as the main project.

---

**Built with**:
- [Chainlit](https://docs.chainlit.io/) - Modern chat UI framework
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) - AI response generation
- [Milvus](https://milvus.io/) - Vector database for clinical guidelines
- Python 3.9+ - Programming language


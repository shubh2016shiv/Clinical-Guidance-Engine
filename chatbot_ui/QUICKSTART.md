# Quick Start Guide - Asclepius Chainlit UI

## Prerequisites Check

Before starting, ensure you have:

1. **Python 3.9+** installed
2. **Milvus** vector database running (with drug data ingested)
3. **Environment variables** configured in `.env` file

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install Chainlit along with all other required packages.

### 2. Verify Environment Configuration

Ensure your `.env` file contains:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o

# Gemini API Configuration (for embeddings)
GEMINI_API_KEY=your_gemini_api_key_here

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=pharmaceutical_drugs

# Application Configuration
ENABLE_STREAMING=true
ENABLE_CLEANUP=false
DEBUG_MODE=false
```

### 3. Verify Milvus is Running

Check that Milvus is accessible:

```bash
# If using Docker
docker ps | grep milvus

# Or check if port 19530 is listening
netstat -an | grep 19530  # Linux/Mac
netstat -an | findstr 19530  # Windows
```

### 4. Verify Drug Data is Loaded

The application requires the pharmaceutical drug database to be ingested into Milvus. If not done:

```bash
cd drug_data_ingestion_pipeline
python milvus_drug_ingestion_pipeline.py
```

### 5. Launch the Chainlit UI

From the project root directory:

**Note**: Port 8000 is typically used by Milvus Attu UI. Use port 8080 to avoid conflicts.

```bash
chainlit run chatbot_ui/app.py --port 8080
```

The application will start and display:

```
Your app is available at http://localhost:8080
```

### 6. Access the Application

Open your web browser and navigate to:

```
http://localhost:8080
```

**Alternative**: If you want to use port 8000 (and Milvus Attu is not running):

```bash
chainlit run chatbot_ui/app.py
# Access at http://localhost:8000
```

You should see:
- Welcome message from Asclepius
- Starter prompts for common medical questions
- Chat interface ready to accept questions

## Usage Examples

### Example 1: Drug Information Query

**User**: "What are the available formulations and dosages for metformin?"

**Expected**: The agent will search the drug database and provide detailed information about metformin formulations with citations from clinical guidelines if available.

### Example 2: Drug Interaction Query

**User**: "Can you explain the drug interactions for warfarin?"

**Expected**: The agent will provide information about warfarin interactions, potentially using both drug database and clinical guidelines.

### Example 3: Clinical Guidelines Query

**User**: "What are the guidelines for treating type 2 diabetes?"

**Expected**: The agent will search clinical guidelines and provide evidence-based recommendations with citations.

## Troubleshooting

### Issue: Application won't start

**Error**: `ModuleNotFoundError: No module named 'chainlit'`

**Solution**:
```bash
pip install chainlit>=2.0.0
```

### Issue: Knowledge base initialization fails

**Error**: "Knowledge base initialization incomplete"

**Possible Causes**:
1. Milvus not running
2. Clinical guidelines files missing
3. Network connectivity issues

**Solution**:
```bash
# Check Milvus status
docker ps | grep milvus

# Verify clinical guidelines directory exists
ls -la clinical_guidelines/

# Check .env configuration
cat .env | grep MILVUS
```

### Issue: No response from agent

**Error**: "Agent not ready"

**Solution**:
1. Refresh the page
2. Check console logs for initialization errors
3. Verify OpenAI API key is valid
4. Check Milvus connection

### Issue: Streaming not working

**Symptom**: Responses appear all at once instead of streaming

**Solution**:
```bash
# Check streaming configuration in .env
echo $ENABLE_STREAMING  # Should be 'true'
```

## Advanced Usage

### Custom Port

```bash
# Recommended: Port 8080 (avoids Milvus Attu conflict)
chainlit run chatbot_ui/app.py --port 8080

# Or use any other available port
chainlit run chatbot_ui/app.py --port 3000
```

### Development Mode (Auto-reload)

```bash
chainlit run chatbot_ui/app.py --watch
```

### Custom Host (Remote Access)

```bash
chainlit run chatbot_ui/app.py --host 0.0.0.0 --port 8000
```

## Conversation Persistence

Conversations are automatically saved to:

```
chatbot_ui/conversations/
```

Each conversation is stored as a JSON file named with the session ID:

```
chatbot_ui/conversations/<session_id>.json
```

### View Conversation History

```bash
# List all conversations
ls -la chatbot_ui/conversations/

# View a specific conversation
cat chatbot_ui/conversations/<session_id>.json | python -m json.tool
```

### Clean Up Old Conversations

You can manually delete old conversation files:

```bash
# Delete conversations older than 30 days
find chatbot_ui/conversations/ -name "*.json" -mtime +30 -delete
```

## Configuration Customization

### Change Welcome Message

Edit `chatbot_ui/config.py`:

```python
WELCOME_MESSAGE = "Your custom welcome message here"
```

### Modify Starter Prompts

Edit `chatbot_ui/starter_prompts.py`:

```python
def get_starter_prompts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Your Custom Prompt",
            message="Your custom message here",
            icon="/public/icon.svg",
        ),
        # Add more starters...
    ]
```

### Adjust Session Timeout

Edit `chatbot_ui/.chainlit`:

```toml
[project]
session_timeout = 1800  # 30 minutes in seconds
```

## Next Steps

1. **Test with Various Queries**: Try different medical questions
2. **Review Conversation Logs**: Check persistence is working
3. **Monitor Performance**: Observe streaming behavior
4. **Customize UI**: Modify theme and starter prompts as needed
5. **Plan MongoDB Migration**: When ready, migrate from JSON to MongoDB

## Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Review logs in the terminal where Chainlit is running
3. Verify all environment variables are set correctly
4. Ensure Milvus is running and accessible
5. Check that drug data has been ingested into Milvus

## Production Deployment

For production deployment:

1. Set `DEBUG_MODE=false` in `.env`
2. Configure authentication (if needed)
3. Use a production-grade database (MongoDB)
4. Set up proper logging and monitoring
5. Configure HTTPS/SSL
6. Implement rate limiting
7. Set up backup for conversation data

---

**Ready to Start**: Run `chainlit run chatbot_ui/app.py --port 8080` and visit `http://localhost:8080`


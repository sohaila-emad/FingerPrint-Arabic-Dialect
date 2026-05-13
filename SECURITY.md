# Security Policy

## API Key Handling

This project uses two external APIs: **ElevenLabs** and **Google Gemini**.

### ✅ How keys are stored (correct way)

Keys are loaded from environment variables via a `.env` file:

```bash
# .env  ← never committed to git
ELEVENLABS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

The `.env` file is listed in `.gitignore` — it will never be committed.
The `.env.example` file (committed) contains placeholder values only.

### ⚠️ Known limitation for demo/local use

The `/config` endpoint in `app.py` returns API keys to the browser so the frontend can call ElevenLabs and Gemini directly:

```python
@app.route('/config')
def config():
    return jsonify({
        "elevenlabs_key": ELEVENLABS_API_KEY,
        "gemini_key":     GEMINI_API_KEY,
    })
```

**This is acceptable for local/demo deployment only.**

For any public deployment, you must:
1. Remove the `/config` endpoint
2. Move all ElevenLabs calls to the backend (`/tts` endpoint)
3. Move all Gemini calls to the backend (`/translate` endpoint)
4. Add authentication and rate limiting

### If you accidentally committed a real API key

1. **Immediately revoke the key** in your provider's dashboard:
   - ElevenLabs: https://elevenlabs.io/app/settings/api-keys
   - Google: https://aistudio.google.com/app/apikey
2. Generate a new key
3. Remove the key from git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch backend/app.py" \
     --prune-empty --tag-name-filter cat -- --all
   ```
   Or use [BFG Repo Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
4. Force push: `git push --force`

> **Note:** Even after removing from history, treat any exposed key as compromised. Always revoke first.

## Reporting a Vulnerability

If you find a security issue in this project, please open a GitHub Issue marked **[SECURITY]**.

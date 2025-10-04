import os
import json
import logging
import uuid
from flask import Flask, request, Response, jsonify, render_template, session, url_for, redirect
from flask_cors import CORS
import requests
import PyPDF2
from io import BytesIO
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Fix 1: Ensure secure secret key from environment variable
app.secret_key = os.environ.get("SESSION_SECRET")
if not app.secret_key and os.environ.get("FLASK_ENV") == "production":
    raise ValueError("SESSION_SECRET environment variable must be set in production")

# Configuration
app.config.update(
    SECRET_KEY=app.secret_key,
    GOOGLE_CLIENT_ID=os.environ.get("GOOGLE_CLIENT_ID"),
    GOOGLE_CLIENT_SECRET=os.environ.get("GOOGLE_CLIENT_SECRET"),
)

# Fix 2: Restrict CORS to specific origins
CORS(app, origins=[os.environ.get("FRONTEND_URL", "http://localhost:3000")])

# Initialize OAuth
oauth = OAuth(app)

# OAuth Registrations - Fixed for Vercel
google = oauth.register(
    name='google',
    client_id=app.config["GOOGLE_CLIENT_ID"],
    client_secret=app.config["GOOGLE_CLIENT_SECRET"],
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={
        'scope': 'openid email profile',
        'token_endpoint_auth_method': 'client_secret_post'
    },
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs'
)

# Token management
TOKEN_LIMIT = 300000

# Fix 4: Store tokens_used in session for user-specific tracking
def get_user_tokens():
    return session.get('tokens_used', 0)

def update_user_tokens(tokens):
    session['tokens_used'] = get_user_tokens() + tokens
    session.modified = True

def count_tokens(text):
    """Approximate token count by splitting on spaces"""
    if not text:
        return 0
    return len(text.split()) + len(text) // 4

# Initialize OpenRouter API key
KEY = os.getenv("OPENROUTER_API_KEY")
# Fix 9: Configurable model
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

# AI Models configuration
MODELS = {
    "logic": {"name": "Logic AI", "description": "analytical, structured, step-by-step"},
    "creative": {"name": "Creative AI", "description": "poetic, metaphorical, emotional"},
    "technical": {"name": "Technical AI", "description": "precise, technical, detail-oriented"},
    "philosophical": {"name": "Philosophical AI", "description": "deep, reflective, abstract"},
    "humorous": {"name": "Humorous AI", "description": "witty, lighthearted, engaging"}
}

SYSTEM_PROMPTS = {
    "logic": "You are Logic AI — analytical, structured, step-by-step. Provide clear, logical reasoning and systematic approaches. Break down complex problems into manageable steps and explain your reasoning clearly.",
    "creative": "You are Creative AI — poetic, metaphorical, emotional. Use imaginative language and creative perspectives. Think outside the box and provide innovative solutions with vivid descriptions.",
    "technical": "You are Technical AI — precise, technical, detail-oriented. Provide accurate, detailed, and technically sound responses, focusing on facts, specifications, and practical applications.",
    "philosophical": "You are Philosophical AI — deep, reflective, abstract. Offer profound insights, explore existential questions, and provide thoughtful, nuanced perspectives.",
    "humorous": "You are Humorous AI — witty, lighthearted, engaging. Deliver responses with humor, clever analogies, and a playful tone while remaining relevant and informative."
}

# Routes
@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)

@app.route('/google_login')
def google_login():
    try:
        # Fix 7: Use environment variable for redirect URI
        base_url = os.environ.get("BASE_URL", "http://localhost:5000")
        redirect_uri = f"{base_url}/login/google/authorize"
        logger.info(f"Starting OAuth with redirect: {redirect_uri}")
        return oauth.google.authorize_redirect(redirect_uri)
    except Exception as e:
        # Fix 6: Return JSON error response
        logger.error(f"Google login error: {str(e)}", exc_info=True)
        return jsonify(error="Authentication failed"), 500

@app.route('/login/google/authorize')
def google_authorize():
    try:
        logger.info("Google authorize endpoint hit")
        token = oauth.google.authorize_access_token()
        logger.info("Token received successfully")
        
        if not token:
            logger.error("No token received from Google")
            return jsonify(error="Authentication failed: No token received"), 400
        
        resp = oauth.google.get('userinfo', token=token)
        logger.info(f"User info response status: {resp.status_code}")
        
        if resp.status_code != 200:
            logger.error(f"Google API error: {resp.status_code} - {resp.text}")
            return jsonify(error=f"Failed to fetch user information: {resp.status_code}"), 400
        
        user_info = resp.json()
        # Fix 10: Avoid logging sensitive information
        logger.info("User info received")
        
        session['user'] = {
            'name': user_info.get('name', 'User'),
            'email': user_info.get('email', ''),
            'picture': user_info.get('picture', ''),
            'provider': 'google'
        }
        
        session.modified = True
        logger.info("User logged in successfully")
        return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Google auth error: {str(e)}", exc_info=True)
        return jsonify(error="Authentication failed"), 400

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('tokens_used', None)  # Fix 4: Reset user-specific tokens
    return redirect(url_for('index'))

# File processing
def extract_text_from_pdf(file_content):
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return None

# AI Generation using direct HTTP requests
def generate(bot_name: str, system: str, user: str, file_contents: list = None):
    if not KEY:
        yield f"data: {json.dumps({'bot': bot_name, 'error': 'OpenRouter API key not configured'})}\n\n"
        return
    
    try:
        full_user_prompt = user
        if file_contents:
            file_context = "\n\n".join(file_contents)
            full_user_prompt = f"{user}\n\nAttached files content:\n{file_context}"
        
        if not session.get('user'):
            yield f"data: {json.dumps({'bot': bot_name, 'error': 'Please login first'})}\n\n"
            return
        
        system_tokens = count_tokens(system)
        user_tokens = count_tokens(full_user_prompt)
        update_user_tokens(system_tokens + user_tokens)  # Fix 4: Update user-specific tokens
        
        payload = {
            "model": OPENROUTER_MODEL,  # Fix 9: Use configurable model
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": full_user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {KEY}",
            "HTTP-Referer": request.host_url,
            "X-Title": "Pentad-Chat",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            error_msg = f"API error: {response.status_code} - {response.text}"
            yield f"data: {json.dumps({'bot': bot_name, 'error': error_msg})}\n\n"
            return
        
        bot_tokens = 0
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data)
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                full_response += content
                                bot_tokens += count_tokens(content)
                                yield f"data: {json.dumps({'bot': bot_name, 'text': content})}\n\n"
                    except json.JSONDecodeError:
                        continue
        
        update_user_tokens(bot_tokens)  # Fix 4: Update user-specific tokens
        yield f"data: {json.dumps({'bot': bot_name, 'done': True, 'tokens': get_user_tokens()})}\n\n"
    
    except Exception as exc:
        logger.error(f"Generation error for {bot_name}: {str(exc)}")
        error_msg = f"Failed to generate response: {str(exc)}"
        yield f"data: {json.dumps({'bot': bot_name, 'error': error_msg})}\n\n"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not session.get('user'):
            return jsonify(error="Please login first"), 401
        
        data = request.json or {}
        prompt = data.get("prompt", "").strip()
        fileUrls = data.get("fileUrls", [])
        
        # Fix 11: Validate prompt length
        if len(prompt) > 5000:
            return jsonify(error="Prompt too long (max 5000 characters)"), 400
        
        if not prompt and not fileUrls:
            return jsonify(error="Empty prompt and no files provided"), 400
        
        if get_user_tokens() >= TOKEN_LIMIT:
            return jsonify(error=f"Token limit reached ({get_user_tokens()}/{TOKEN_LIMIT})"), 429
        
        # Fix 8: Extract file contents if URLs point to accessible files
        file_contents = []
        if fileUrls:
            for file_url in fileUrls:
                try:
                    response = requests.get(file_url, timeout=10)
                    if response.status_code == 200 and file_url.lower().endswith('.pdf'):
                        content = extract_text_from_pdf(response.content)
                        if content:
                            file_contents.append(content)
                        else:
                            file_contents.append(f"File at {file_url}: Unable to extract content")
                    else:
                        file_contents.append(f"File at {file_url}: Unable to access or not a PDF")
                except Exception as e:
                    logger.error(f"Error fetching file {file_url}: {str(e)}")
                    file_contents.append(f"File at {file_url}: Error accessing file")

        def event_stream():
            generators = {}
            for key in MODELS.keys():
                generators[key] = generate(key, SYSTEM_PROMPTS[key], prompt, file_contents)
            
            active_bots = list(MODELS.keys())
            
            while active_bots:
                for bot_name in active_bots[:]:
                    try:
                        chunk = next(generators[bot_name])
                        yield chunk
                        
                        try:
                            chunk_data = json.loads(chunk.split('data: ')[1])
                            if chunk_data.get('done') or chunk_data.get('error'):
                                active_bots.remove(bot_name)
                        except:
                            pass
                        
                    except StopIteration:
                        active_bots.remove(bot_name)
                    except Exception as e:
                        logger.error(f"Stream error for {bot_name}: {str(e)}")
                        active_bots.remove(bot_name)
            
            yield f"data: {json.dumps({'all_done': True, 'tokens': get_user_tokens()})}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            },
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify(error="Internal server error"), 500

@app.route("/asklurk", methods=["POST"])
def asklurk():
    try:
        if not session.get('user'):
            return jsonify(best="", error="Please login first"), 401
        
        data = request.json or {}
        answers = data.get("answers", {})
        prompt = data.get("prompt", "")
        
        # Fix 11: Validate input
        if len(prompt) > 5000:
            return jsonify(best="", error="Prompt too long (max 5000 characters)"), 400
        
        if not answers:
            return jsonify(best="", error="No responses to analyze"), 400
        
        if not KEY:
            return jsonify(best="", error="OpenRouter API key not configured"), 500
        
        try:
            merged_content = f"Original question: {prompt}\n\n"
            for key, response in answers.items():
                if key in MODELS:
                    merged_content += f"## {MODELS[key]['name']}:\n{response}\n\n"
            
            payload = {
                "model": OPENROUTER_MODEL,  # Fix 9: Use configurable model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are AskLurk - an expert AI synthesizer. Your task is to analyze responses from Logic AI, Creative AI, Technical AI, Philosophical AI, and Humorous AI to create the single best answer. Combine the logical reasoning, creative insights, technical accuracy, philosophical depth, and humorous engagement to provide a comprehensive, well-structured response that leverages the strengths of all approaches. Structure your response to be insightful, engaging, and balanced."
                    },
                    {
                        "role": "user",
                        "content": f"Please analyze these AI responses to the question: \"{prompt}\"\n\nHere are the responses:\n{merged_content}\n\nPlease provide the best synthesized answer that leverages the strengths of all AI responses:"
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1500,
            }
            
            headers = {
                "Authorization": f"Bearer {KEY}",
                "HTTP-Referer": request.host_url,
                "X-Title": "Pentad-Chat",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            best_answer = result['choices'][0]['message']['content']
            asklurk_tokens = count_tokens(best_answer)
            update_user_tokens(asklurk_tokens)  # Fix 4: Update user-specific tokens
            
            return jsonify(best=best_answer, tokens_used=get_user_tokens())
        
        except Exception as e:
            logger.error(f"AskLurk error: {str(e)}")
            if answers:
                first_response = next(iter(answers.values()))
                return jsonify(best=f"Fallback - Using first response:\n\n{first_response}", error="AI synthesis failed")
            return jsonify(best="", error="No responses available for synthesis")
    
    except Exception as e:
        logger.error(f"AskLurk error: {str(e)}")
        return jsonify(error="Internal server error"), 500

@app.route("/upload", methods=["POST"])
def upload():
    """File upload endpoint - simplified for Vercel"""
    try:
        if 'files' not in request.files:
            return jsonify(urls=[], error="No files provided"), 400
        
        files = request.files.getlist('files')
        urls = []
        
        # Fix 5: Validate file types and sizes
        ALLOWED_EXTENSIONS = {'pdf'}
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        
        for file in files:
            if file.filename == '':
                continue
            
            # Validate file extension
            if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                continue
            
            # Validate file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            if file_size > MAX_FILE_SIZE:
                continue
            file.seek(0)
            
            # In Vercel, use in-memory processing
            name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            urls.append(f"/static/uploads/{name}")
            # Note: For persistent storage, integrate with S3 or similar
        
        return jsonify(urls=urls)
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify(error="File upload failed"), 500

@app.route("/tokens", methods=["GET"])
def get_tokens():
    return jsonify({
        "tokens_used": get_user_tokens(),
        "token_limit": TOKEN_LIMIT,
        "remaining_tokens": TOKEN_LIMIT - get_user_tokens(),
        "usage_percentage": (get_user_tokens() / TOKEN_LIMIT) * 100
    })

@app.route("/reset-tokens", methods=["POST"])
def reset_tokens():
    session['tokens_used'] = 0  # Fix 4: Reset user-specific tokens
    session.modified = True
    return jsonify({"message": "Token counter reset", "tokens_used": get_user_tokens()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "api_key_configured": bool(KEY),
        "models_configured": len(MODELS),
        "tokens_used": get_user_tokens()
    })

# Vercel compatibility
def create_app():
    return app

import logging
import os
import re
import traceback
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YouTube Transcript Retriever - Groq RAG",
    page_icon="🎥",
    layout="wide",
)


# =========================================================
# LOGGING SETUP
# =========================================================

LOG_FILE = "youtube_chatbot.log"


def setup_logging():
    """
    Configure application logging.

    Logs are written to:
    1. Terminal / PowerShell
    2. youtube_chatbot.log
    """

    logger = logging.getLogger("youtube_chatbot")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -----------------------------------------------------
    # File handler
    # -----------------------------------------------------

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # -----------------------------------------------------
    # Console handler
    # -----------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # -----------------------------------------------------
    # Add handlers
    # -----------------------------------------------------

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


logger = setup_logging()


# =========================================================
# STREAMLIT LOG STORAGE
# =========================================================

if "logs" not in st.session_state:
    st.session_state["logs"] = []


def log_message(level, message):
    """
    Log message to terminal, file and Streamlit UI.
    """

    timestamp = datetime.now().strftime("%H:%M:%S")

    ui_message = f"[{timestamp}] {level.upper()} | {message}"

    st.session_state["logs"].append(ui_message)

    if level == "debug":
        logger.debug(message)

    elif level == "info":
        logger.info(message)

    elif level == "warning":
        logger.warning(message)

    elif level == "error":
        logger.error(message)

    else:
        logger.info(message)


def log_exception(message):
    """
    Log exception with complete traceback.
    """

    logger.error(message)
    logger.error(traceback.format_exc())

    st.session_state["logs"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] ERROR | {message}"
    )

    st.session_state["logs"].append(
        traceback.format_exc()
    )


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

log_message("info", "Starting YouTube Transcript Retriever application.")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:

    log_message(
        "info",
        "GROQ_API_KEY loaded successfully."
    )

else:

    log_message(
        "error",
        "GROQ_API_KEY is missing from the .env file."
    )

    st.error(
        "GROQ_API_KEY is missing from the .env file."
    )

    st.stop()


# =========================================================
# YOUTUBE VIDEO ID EXTRACTION
# =========================================================

def extract_video_id(url):
    """
    Extract YouTube video ID without relying on pytube.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """

    log_message(
        "info",
        f"Starting YouTube URL parsing: {url}"
    )

    if not url:
        log_message(
            "error",
            "Received empty YouTube URL."
        )
        return None

    url = url.strip()

    # -----------------------------------------------------
    # Standard YouTube URL
    # -----------------------------------------------------

    try:

        parsed = urlparse(url)

        hostname = parsed.hostname or ""

        hostname = hostname.lower()

        log_message(
            "debug",
            f"Parsed hostname: {hostname}"
        )

        # youtube.com/watch?v=...
        if "youtube.com" in hostname:

            query_params = parse_qs(parsed.query)

            if "v" in query_params:

                video_id = query_params["v"][0]

                if video_id:

                    log_message(
                        "info",
                        f"Video ID extracted from watch URL: {video_id}"
                    )

                    return video_id

            # /shorts/VIDEO_ID
            shorts_match = re.search(
                r"/shorts/([A-Za-z0-9_-]{11})",
                parsed.path,
            )

            if shorts_match:

                video_id = shorts_match.group(1)

                log_message(
                    "info",
                    f"Video ID extracted from Shorts URL: {video_id}"
                )

                return video_id

            # /embed/VIDEO_ID
            embed_match = re.search(
                r"/embed/([A-Za-z0-9_-]{11})",
                parsed.path,
            )

            if embed_match:

                video_id = embed_match.group(1)

                log_message(
                    "info",
                    f"Video ID extracted from embed URL: {video_id}"
                )

                return video_id

        # -------------------------------------------------
        # youtu.be/VIDEO_ID
        # -------------------------------------------------

        if "youtu.be" in hostname:

            video_id = parsed.path.strip("/").split("/")[0]

            if video_id:

                # Remove anything after whitespace
                video_id = video_id.split()[0]

                # Validate basic YouTube ID format
                if re.fullmatch(
                    r"[A-Za-z0-9_-]{11}",
                    video_id,
                ):

                    log_message(
                        "info",
                        f"Video ID extracted from youtu.be URL: {video_id}"
                    )

                    return video_id

    except Exception:

        log_exception(
            "Exception occurred while parsing YouTube URL."
        )

        return None

    # -----------------------------------------------------
    # Final fallback regex
    # -----------------------------------------------------

    regex_patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in regex_patterns:

        match = re.search(
            pattern,
            url,
        )

        if match:

            video_id = match.group(1)

            log_message(
                "info",
                f"Video ID extracted using fallback regex: {video_id}"
            )

            return video_id

    log_message(
        "error",
        "Unable to extract a valid YouTube video ID."
    )

    return None


# =========================================================
# TRANSCRIPT ITEM TEXT EXTRACTION
# =========================================================

def extract_text_from_transcript_items(transcript):
    """
    Convert transcript snippets to plain text.

    Works with object-style snippets and dictionary-style snippets.
    """

    log_message(
        "debug",
        "Converting transcript snippets into plain text."
    )

    text_parts = []

    for item in transcript:

        try:

            if hasattr(item, "text"):

                text = item.text

            elif isinstance(item, dict):

                text = item.get("text", "")

            else:

                text = str(item)

            if text:

                text_parts.append(text)

        except Exception:

            log_message(
                "warning",
                f"Could not parse transcript item: {item}"
            )

    result = " ".join(text_parts).strip()

    log_message(
        "info",
        f"Transcript text assembled. Character count: {len(result)}"
    )

    return result


# =========================================================
# LIST AVAILABLE TRANSCRIPTS
# =========================================================

def inspect_available_transcripts(api, video_id):
    """
    Retrieve and log metadata about available transcripts.
    """

    log_message(
        "info",
        "Checking all available transcripts..."
    )

    try:

        transcript_list = api.list(video_id)

        available_transcripts = []

        for transcript in transcript_list:

            language = getattr(
                transcript,
                "language",
                "Unknown",
            )

            language_code = getattr(
                transcript,
                "language_code",
                "Unknown",
            )

            is_generated = getattr(
                transcript,
                "is_generated",
                "Unknown",
            )

            is_translatable = getattr(
                transcript,
                "is_translatable",
                "Unknown",
            )

            available_transcripts.append(
                {
                    "language": language,
                    "language_code": language_code,
                    "is_generated": is_generated,
                    "is_translatable": is_translatable,
                }
            )

            log_message(
                "info",
                (
                    f"Transcript found | "
                    f"language={language} | "
                    f"code={language_code} | "
                    f"generated={is_generated} | "
                    f"translatable={is_translatable}"
                )
            )

        if not available_transcripts:

            log_message(
                "warning",
                "Transcript list returned no entries."
            )

        else:

            log_message(
                "info",
                f"Total available transcripts: {len(available_transcripts)}"
            )

        return transcript_list, available_transcripts

    except TranscriptsDisabled:

        log_message(
            "error",
            "YouTube reports that transcripts are disabled."
        )

        return None, []

    except NoTranscriptFound:

        log_message(
            "warning",
            "No transcripts were found for this video."
        )

        return None, []

    except Exception:

        log_exception(
            "Unexpected error while listing transcripts."
        )

        return None, []


# =========================================================
# FETCH YOUTUBE TRANSCRIPT
# =========================================================

def get_youtube_transcript(url):
    """
    Fetch YouTube transcript.

    Strategy:
    1. Extract video ID.
    2. Inspect all available transcripts.
    3. Prefer English.
    4. Fall back to first available transcript.
    5. Convert transcript to plain text.
    """

    log_message(
        "info",
        "=================================================="
    )

    log_message(
        "info",
        "STARTING TRANSCRIPT RETRIEVAL"
    )

    log_message(
        "info",
        "=================================================="
    )

    # -----------------------------------------------------
    # Extract video ID
    # -----------------------------------------------------

    video_id = extract_video_id(url)

    if not video_id:

        st.error(
            "Could not extract a valid YouTube video ID."
        )

        return None

    log_message(
        "info",
        f"Using YouTube video ID: {video_id}"
    )

    # -----------------------------------------------------
    # Create transcript API client
    # -----------------------------------------------------

    try:

        log_message(
            "info",
            "Creating YouTubeTranscriptApi client..."
        )

        api = YouTubeTranscriptApi()

        log_message(
            "info",
            "YouTubeTranscriptApi client created successfully."
        )

    except Exception:

        log_exception(
            "Failed to initialize YouTubeTranscriptApi."
        )

        return None

    # -----------------------------------------------------
    # First attempt: direct English fetch
    # -----------------------------------------------------

    log_message(
        "info",
        "Attempting direct English transcript retrieval..."
    )

    try:

        transcript = api.fetch(
            video_id,
            languages=[
                "en",
                "en-US",
                "en-GB",
            ],
        )

        log_message(
            "info",
            "English transcript retrieved successfully."
        )

        transcript_data = transcript

        text = extract_text_from_transcript_items(
            transcript_data
        )

        if text:

            log_message(
                "info",
                "Direct English transcript contains usable text."
            )

            log_message(
                "info",
                f"Final transcript length: {len(text)} characters."
            )

            return text

        log_message(
            "warning",
            "Direct English transcript returned empty text."
        )

    except NoTranscriptFound:

        log_message(
            "warning",
            "English transcript not found. Inspecting available languages..."
        )

    except TranscriptsDisabled:

        log_message(
            "error",
            "Transcripts are disabled for this video."
        )

        st.error(
            "❌ Transcripts are disabled for this video."
        )

        return None

    except VideoUnavailable:

        log_message(
            "error",
            "Video is unavailable."
        )

        st.error(
            "❌ This YouTube video is unavailable."
        )

        return None

    except CouldNotRetrieveTranscript as error:

        log_message(
            "warning",
            f"Could not directly retrieve English transcript: {error}"
        )

    except Exception:

        log_exception(
            "Unexpected error during direct transcript retrieval."
        )

    # -----------------------------------------------------
    # Inspect available transcripts
    # -----------------------------------------------------

    transcript_list, available_transcripts = (
        inspect_available_transcripts(
            api,
            video_id,
        )
    )

    if transcript_list is None:

        st.error(
            "❌ No transcript could be retrieved for this video."
        )

        log_message(
            "error",
            "Transcript list could not be retrieved."
        )

        return None

    # -----------------------------------------------------
    # Find English transcript
    # -----------------------------------------------------

    selected_transcript = None

    log_message(
        "info",
        "Searching for English transcript..."
    )

    try:

        selected_transcript = (
            transcript_list.find_transcript(
                [
                    "en",
                    "en-US",
                    "en-GB",
                ]
            )
        )

        log_message(
            "info",
            (
                "English transcript selected: "
                f"{selected_transcript.language} "
                f"({selected_transcript.language_code})"
            )
        )

    except NoTranscriptFound:

        log_message(
            "warning",
            "No English transcript available."
        )

    except Exception:

        log_exception(
            "Error while searching for English transcript."
        )

    # -----------------------------------------------------
    # Fallback to first available transcript
    # -----------------------------------------------------

    if selected_transcript is None:

        log_message(
            "warning",
            "Falling back to first available transcript."
        )

        try:

            selected_transcript = next(
                iter(transcript_list)
            )

            log_message(
                "info",
                (
                    "Fallback transcript selected: "
                    f"{selected_transcript.language} "
                    f"({selected_transcript.language_code})"
                )
            )

        except StopIteration:

            log_message(
                "error",
                "Transcript list is empty."
            )

            st.error(
                "❌ No usable transcript is available."
            )

            return None

        except Exception:

            log_exception(
                "Failed to select fallback transcript."
            )

            return None

    # -----------------------------------------------------
    # Fetch selected transcript
    # -----------------------------------------------------

    try:

        log_message(
            "info",
            "Fetching selected transcript..."
        )

        fetched_transcript = (
            selected_transcript.fetch()
        )

        log_message(
            "info",
            "Selected transcript fetched successfully."
        )

        text = extract_text_from_transcript_items(
            fetched_transcript
        )

        if not text:

            log_message(
                "error",
                "Fetched transcript contains no text."
            )

            st.error(
                "❌ Transcript was retrieved but contains no text."
            )

            return None

        log_message(
            "info",
            f"Final transcript length: {len(text)} characters."
        )

        # -------------------------------------------------
        # Preview for logs
        # -------------------------------------------------

        preview = text[:500].replace(
            "\n",
            " ",
        )

        log_message(
            "debug",
            f"Transcript preview: {preview}"
        )

        log_message(
            "info",
            "Transcript retrieval completed successfully."
        )

        return text

    except TranscriptsDisabled:

        log_message(
            "error",
            "Transcripts are disabled."
        )

        st.error(
            "❌ Transcripts are disabled for this video."
        )

    except NoTranscriptFound:

        log_message(
            "error",
            "No transcript found after fallback selection."
        )

        st.error(
            "❌ No transcript could be retrieved."
        )

    except VideoUnavailable:

        log_message(
            "error",
            "Video became unavailable while retrieving transcript."
        )

        st.error(
            "❌ Video is unavailable."
        )

    except CouldNotRetrieveTranscript as error:

        log_message(
            "error",
            f"Could not retrieve transcript: {error}"
        )

        st.error(
            f"❌ Could not retrieve transcript: {error}"
        )

    except Exception:

        log_exception(
            "Unexpected error while fetching selected transcript."
        )

        st.error(
            "❌ An unexpected error occurred while retrieving the transcript."
        )

    return None


# =========================================================
# CREATE RAG QA CHAIN
# =========================================================

def create_qa_chain(transcript_text):
    """
    Create:

    Transcript
        ↓
    Text chunks
        ↓
    HuggingFace embeddings
        ↓
    FAISS vector store
        ↓
    Groq LLM
        ↓
    RetrievalQA
    """

    log_message(
        "info",
        "=================================================="
    )

    log_message(
        "info",
        "STARTING RAG PIPELINE"
    )

    log_message(
        "info",
        "=================================================="
    )

    # -----------------------------------------------------
    # Validate transcript
    # -----------------------------------------------------

    if not transcript_text:

        raise ValueError(
            "Transcript text is empty."
        )

    log_message(
        "info",
        f"Transcript received: {len(transcript_text)} characters."
    )

    # -----------------------------------------------------
    # Save transcript
    # -----------------------------------------------------

    transcript_file = "transcript.txt"

    log_message(
        "info",
        f"Saving transcript to {transcript_file}..."
    )

    with open(
        transcript_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            transcript_text
        )

    log_message(
        "info",
        "Transcript saved successfully."
    )

    # -----------------------------------------------------
    # Load transcript
    # -----------------------------------------------------

    log_message(
        "info",
        "Loading transcript using TextLoader..."
    )

    loader = TextLoader(
        transcript_file,
        encoding="utf-8",
    )

    documents = loader.load()

    log_message(
        "info",
        f"Loaded documents: {len(documents)}"
    )

    if not documents:

        raise ValueError(
            "TextLoader returned no documents."
        )

    # -----------------------------------------------------
    # Text splitting
    # -----------------------------------------------------

    log_message(
        "info",
        "Splitting transcript into chunks..."
    )

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(
        documents
    )

    log_message(
        "info",
        f"Created {len(chunks)} transcript chunks."
    )

    if not chunks:

        raise ValueError(
            "No transcript chunks were created."
        )

    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------

    log_message(
        "info",
        "Initializing HuggingFace embeddings..."
    )

    embedding_model = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    log_message(
        "info",
        f"Embedding model: {embedding_model}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    log_message(
        "info",
        "HuggingFace embeddings initialized successfully."
    )

    # -----------------------------------------------------
    # FAISS
    # -----------------------------------------------------

    log_message(
        "info",
        "Creating FAISS vector store..."
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    log_message(
        "info",
        "FAISS vector store created successfully."
    )

    # -----------------------------------------------------
    # Groq
    # -----------------------------------------------------

    log_message(
        "info",
        "Initializing Groq LLM..."
    )

    groq_model = "openai/gpt-oss-20b"

    log_message(
        "info",
        f"Groq model: {groq_model}"
    )

    llm = ChatGroq(
        model=groq_model,
        temperature=0,
        groq_api_key=GROQ_API_KEY,
    )

    log_message(
        "info",
        "Groq LLM initialized successfully."
    )

    # -----------------------------------------------------
    # RetrievalQA
    # -----------------------------------------------------

    log_message(
        "info",
        "Creating RetrievalQA chain..."
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 4
            }
        ),
        return_source_documents=False,
    )

    log_message(
        "info",
        "RetrievalQA chain created successfully."
    )

    log_message(
        "info",
        "RAG pipeline completed successfully."
    )

    return qa_chain


# =========================================================
# UI HEADER
# =========================================================

st.title(
    "🎥 YouTube Transcript Retriever - Groq RAG"
)

st.write(
    "Enter a YouTube video URL, process its transcript, "
    "and ask questions about the video."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Debug Information")

    st.write(
        f"Python logging file: `{LOG_FILE}`"
    )

    if st.button("Clear UI Logs"):

        st.session_state["logs"] = []

        st.rerun()

    st.divider()

    st.write(
        "### Runtime"
    )

    st.write(
        f"Python executable:"
    )

    st.code(
        os.sys.executable
    )

    st.write(
        f"Working directory:"
    )

    st.code(
        os.getcwd()
    )


# =========================================================
# YOUTUBE URL INPUT
# =========================================================

video_url = st.text_input(
    "Enter YouTube video URL:",
    placeholder=(
        "https://www.youtube.com/watch?v=..."
    ),
)


# =========================================================
# PROCESS VIDEO
# =========================================================

if st.button(
    "🚀 Process Video",
    type="primary",
):

    # Clear previous logs for this processing attempt
    st.session_state["logs"] = []

    log_message(
        "info",
        "User clicked 'Process Video'."
    )

    if not video_url.strip():

        log_message(
            "warning",
            "User submitted an empty URL."
        )

        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        with st.spinner(
            "Processing YouTube video..."
        ):

            try:

                transcript_text = (
                    get_youtube_transcript(
                        video_url
                    )
                )

                # -------------------------------------------------
                # Transcript successfully retrieved
                # -------------------------------------------------

                if transcript_text:

                    log_message(
                        "info",
                        "Transcript successfully retrieved."
                    )

                    # -------------------------------------------------
                    # Create QA chain
                    # -------------------------------------------------

                    qa_chain = (
                        create_qa_chain(
                            transcript_text
                        )
                    )

                    # Store in session
                    st.session_state[
                        "qa_chain"
                    ] = qa_chain

                    st.session_state[
                        "video_processed"
                    ] = True

                    st.session_state[
                        "transcript_length"
                    ] = len(
                        transcript_text
                    )

                    log_message(
                        "info",
                        "QA chain stored in Streamlit session."
                    )

                    log_message(
                        "info",
                        "VIDEO PROCESSING COMPLETED SUCCESSFULLY."
                    )

                    st.success(
                        "✅ Transcript processed successfully. "
                        "You can now ask questions."
                    )

                    # -------------------------------------------------
                    # Transcript information
                    # -------------------------------------------------

                    st.info(
                        f"Transcript size: "
                        f"{len(transcript_text):,} characters"
                    )

                else:

                    log_message(
                        "error",
                        "Transcript retrieval returned None."
                    )

                    st.error(
                        "❌ Failed to retrieve transcript."
                    )

            except Exception as error:

                log_exception(
                    f"Fatal error during video processing: {error}"
                )

                st.error(
                    f"❌ Processing failed: {error}"
                )


# =========================================================
# QUESTION ANSWERING
# =========================================================

if "qa_chain" in st.session_state:

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    user_question = st.text_input(
        "Ask a question about the transcript:",
        placeholder=(
            "What is this video about?"
        ),
    )

    if user_question.strip():

        log_message(
            "info",
            f"User question received: {user_question}"
        )

        with st.spinner(
            "Generating answer..."
        ):

            try:

                # -------------------------------------------------
                # Invoke RetrievalQA
                # -------------------------------------------------

                log_message(
                    "info",
                    "Sending question to RetrievalQA..."
                )

                answer = (
                    st.session_state[
                        "qa_chain"
                    ].invoke(
                        {
                            "query": user_question
                        }
                    )
                )

                log_message(
                    "info",
                    "RetrievalQA returned successfully."
                )

                # -------------------------------------------------
                # Extract answer
                # -------------------------------------------------

                result = answer.get(
                    "result",
                    "",
                )

                if not result:

                    log_message(
                        "warning",
                        "QA response did not contain a 'result' field."
                    )

                    st.warning(
                        "The model returned an empty answer."
                    )

                else:

                    log_message(
                        "info",
                        f"Generated answer length: {len(result)} characters."
                    )

                    log_message(
                        "debug",
                        f"Generated answer: {result}"
                    )

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(
                        result
                    )

            except Exception as error:

                log_exception(
                    f"Failed to generate answer: {error}"
                )

                st.error(
                    f"❌ Failed to generate answer: {error}"
                )


# =========================================================
# LOG DISPLAY
# =========================================================

st.divider()

st.subheader(
    "🧾 Application Logs"
)

if st.session_state["logs"]:

    with st.expander(
        "Show detailed logs",
        expanded=False,
    ):

        st.code(
            "\n".join(
                st.session_state["logs"]
            ),
            language="text",
        )

else:

    st.caption(
        "No logs generated yet."
    )


# =========================================================
# LOG FILE INFORMATION
# =========================================================

if os.path.exists(LOG_FILE):

    file_size = os.path.getsize(
        LOG_FILE
    )

    st.caption(
        f"Full log file: `{os.path.abspath(LOG_FILE)}` "
        f"({file_size:,} bytes)"
    )
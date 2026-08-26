import json
import logging
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import types


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)


# =========================================================
# CONFIGURATION
# =========================================================

LOG_FILE = "youtube_chatbot.log"

TRANSCRIPT_FILENAME = "transcript.txt"
CHUNKS_FILENAME = "chunks.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
TOKEN_USAGE_FILENAME = "token_usage.json"
PROCESSING_LOG_FILENAME = "processing.log"

# Current Gemini model being used by your working app.
GEMINI_MODEL = "gemini-3.6-flash"

# Local embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG configuration.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 6

# Root directory for generated RAG data.
RAG_ROOT = Path("rag_data")


# =========================================================
# GLOBAL LOGGING
# =========================================================

def setup_logging():
    logger = logging.getLogger("youtube_rag")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler()

    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


logger = setup_logging()


# =========================================================
# STREAMLIT LOG STATE
# =========================================================

if "logs" not in st.session_state:
    st.session_state["logs"] = []


def log(level, message):
    """
    Log to:
    - terminal
    - youtube_chatbot.log
    - Streamlit UI
    """

    timestamp = datetime.now().strftime("%H:%M:%S")

    display_message = (
        f"[{timestamp}] "
        f"{level.upper()} | "
        f"{message}"
    )

    st.session_state["logs"].append(display_message)

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


def log_exception(message, local_logger=None):
    """
    Log complete exception traceback.
    """

    target_logger = local_logger or logger

    target_logger.error(message)
    target_logger.error(traceback.format_exc())

    timestamp = datetime.now().strftime("%H:%M:%S")

    st.session_state["logs"].append(
        f"[{timestamp}] ERROR | {message}"
    )

    st.session_state["logs"].append(
        traceback.format_exc()
    )


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    log(
        "error",
        "GEMINI_API_KEY is missing from .env",
    )

    st.error(
        "GEMINI_API_KEY is missing from your .env file."
    )

    st.stop()


log(
    "info",
    "GEMINI_API_KEY loaded successfully.",
)


# =========================================================
# GEMINI CLIENT
# =========================================================

try:

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    log(
        "info",
        (
            "Gemini client initialized successfully. "
            f"Model={GEMINI_MODEL}"
        ),
    )

except Exception as error:

    log_exception(
        f"Failed to initialize Gemini client: {error}"
    )

    st.error(
        f"Failed to initialize Gemini client: {error}"
    )

    st.stop()


# =========================================================
# OPTIONAL GEMINI MODEL VALIDATION
# =========================================================

def validate_gemini_model():
    """
    Attempt to verify that the configured Gemini model
    is visible to the current API key.

    Some API configurations may expose model aliases
    differently, so failure of this diagnostic should not
    automatically stop the application.
    """

    try:

        log(
            "info",
            f"Checking Gemini model availability: {GEMINI_MODEL}",
        )

        models = list(
            gemini_client.models.list()
        )

        model_names = []

        for model in models:

            name = getattr(
                model,
                "name",
                None,
            )

            if name:

                model_names.append(
                    name.replace(
                        "models/",
                        "",
                    )
                )

        log(
            "info",
            (
                f"Gemini ModelService returned "
                f"{len(model_names)} models."
            ),
        )

        if GEMINI_MODEL in model_names:

            log(
                "info",
                f"Gemini model confirmed: {GEMINI_MODEL}",
            )

        else:

            log(
                "warning",
                (
                    f"Configured model {GEMINI_MODEL} "
                    "was not found in the returned model list. "
                    "The model may still be available through "
                    "the current endpoint."
                ),
            )

    except Exception as error:

        log(
            "warning",
            (
                "Gemini model validation could not be completed: "
                f"{error}"
            ),
        )


validate_gemini_model()


# =========================================================
# CREATE PER-VIDEO LOGGER
# =========================================================

def create_video_logger(video_id):
    """
    Creates:

    rag_data/<video_id>/processing.log
    """

    video_dir = (
        RAG_ROOT / video_id
    )

    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        video_dir / PROCESSING_LOG_FILENAME
    )

    video_logger_name = (
        f"youtube_rag.video.{video_id}"
    )

    video_logger = logging.getLogger(
        video_logger_name
    )

    if video_logger.handlers:
        return video_logger

    video_logger.setLevel(
        logging.DEBUG
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.DEBUG
    )

    file_handler.setFormatter(
        formatter
    )

    video_logger.addHandler(
        file_handler
    )

    video_logger.propagate = False

    return video_logger


# =========================================================
# YOUTUBE VIDEO ID
# =========================================================

def extract_video_id(url):
    log(
        "info",
        f"Parsing YouTube URL: {url}",
    )

    if not url:
        return None

    url = url.strip()

    try:

        parsed = urlparse(url)

        hostname = (
            parsed.hostname or ""
        ).lower()

        log(
            "debug",
            f"Parsed hostname: {hostname}",
        )

        # -------------------------------------------------
        # Standard youtube.com/watch?v=
        # -------------------------------------------------

        if "youtube.com" in hostname:

            query = parse_qs(
                parsed.query
            )

            if "v" in query:

                video_id = query["v"][0]

                if re.fullmatch(
                    r"[A-Za-z0-9_-]{11}",
                    video_id,
                ):

                    log(
                        "info",
                        (
                            "Extracted video ID from "
                            f"watch URL: {video_id}"
                        ),
                    )

                    return video_id

            # ---------------------------------------------
            # Shorts
            # ---------------------------------------------

            shorts_match = re.search(
                r"/shorts/([A-Za-z0-9_-]{11})",
                parsed.path,
            )

            if shorts_match:

                video_id = (
                    shorts_match.group(1)
                )

                log(
                    "info",
                    (
                        "Extracted Shorts video ID: "
                        f"{video_id}"
                    ),
                )

                return video_id

            # ---------------------------------------------
            # Embed
            # ---------------------------------------------

            embed_match = re.search(
                r"/embed/([A-Za-z0-9_-]{11})",
                parsed.path,
            )

            if embed_match:

                video_id = (
                    embed_match.group(1)
                )

                log(
                    "info",
                    (
                        "Extracted embed video ID: "
                        f"{video_id}"
                    ),
                )

                return video_id

        # -------------------------------------------------
        # youtu.be
        # -------------------------------------------------

        if "youtu.be" in hostname:

            video_id = (
                parsed.path
                .strip("/")
                .split("/")[0]
            )

            if re.fullmatch(
                r"[A-Za-z0-9_-]{11}",
                video_id,
            ):

                log(
                    "info",
                    (
                        "Extracted youtu.be video ID: "
                        f"{video_id}"
                    ),
                )

                return video_id

    except Exception:

        log_exception(
            "Error while parsing YouTube URL."
        )

    log(
        "error",
        "Could not extract a valid YouTube video ID.",
    )

    return None


# =========================================================
# TRANSCRIPT → TEXT
# =========================================================

def transcript_to_text(transcript):
    log(
        "debug",
        "Converting transcript snippets to plain text.",
    )

    parts = []

    for item in transcript:

        try:

            if hasattr(
                item,
                "text",
            ):

                text = item.text

            elif isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text",
                    "",
                )

            else:

                text = str(item)

            if text:
                parts.append(text)

        except Exception:

            log(
                "warning",
                (
                    "Could not parse transcript item: "
                    f"{item}"
                ),
            )

    result = (
        " ".join(parts)
        .strip()
    )

    log(
        "info",
        (
            "Transcript converted successfully. "
            f"Characters={len(result):,}"
        ),
    )

    return result


# =========================================================
# GET YOUTUBE TRANSCRIPT
# =========================================================

def get_youtube_transcript(
    url,
    video_logger=None,
):

    log(
        "info",
        "==================================================",
    )

    log(
        "info",
        "STARTING TRANSCRIPT RETRIEVAL",
    )

    log(
        "info",
        "==================================================",
    )

    if video_logger:
        video_logger.info(
            "STARTING TRANSCRIPT RETRIEVAL"
        )

    video_id = extract_video_id(url)

    if not video_id:

        st.error(
            "Invalid YouTube URL."
        )

        return None

    try:

        api = YouTubeTranscriptApi()

        log(
            "info",
            "YouTubeTranscriptApi initialized.",
        )

    except Exception:

        log_exception(
            "Failed to initialize YouTubeTranscriptApi.",
            video_logger,
        )

        return None

    # -----------------------------------------------------
    # Direct English transcript
    # -----------------------------------------------------

    log(
        "info",
        "Attempting English transcript retrieval.",
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

        text = transcript_to_text(
            transcript
        )

        if text:

            log(
                "info",
                "English transcript retrieved successfully.",
            )

            return text

    except NoTranscriptFound:

        log(
            "warning",
            "No English transcript available.",
        )

    except TranscriptsDisabled:

        log(
            "error",
            "Transcripts are disabled.",
        )

        st.error(
            "Transcripts are disabled for this video."
        )

        return None

    except VideoUnavailable:

        log(
            "error",
            "Video is unavailable.",
        )

        st.error(
            "This YouTube video is unavailable."
        )

        return None

    except CouldNotRetrieveTranscript as error:

        log(
            "warning",
            (
                "English transcript retrieval failed: "
                f"{error}"
            ),
        )

    except Exception:

        log_exception(
            (
                "Unexpected error during "
                "English transcript retrieval."
            ),
            video_logger,
        )

    # -----------------------------------------------------
    # List available transcripts
    # -----------------------------------------------------

    log(
        "info",
        "Checking available transcript languages.",
    )

    try:

        transcript_list = api.list(
            video_id
        )

        available_count = 0

        for transcript in transcript_list:

            language = getattr(
                transcript,
                "language",
                "Unknown",
            )

            code = getattr(
                transcript,
                "language_code",
                "Unknown",
            )

            generated = getattr(
                transcript,
                "is_generated",
                False,
            )

            translatable = getattr(
                transcript,
                "is_translatable",
                False,
            )

            available_count += 1

            log(
                "info",
                (
                    f"Transcript found | "
                    f"language={language} | "
                    f"code={code} | "
                    f"generated={generated} | "
                    f"translatable={translatable}"
                ),
            )

        log(
            "info",
            (
                "Total transcripts available: "
                f"{available_count}"
            ),
        )

    except NoTranscriptFound:

        log(
            "error",
            "No transcripts exist for this video.",
        )

        st.error(
            "No transcript is available for this video."
        )

        return None

    except TranscriptsDisabled:

        log(
            "error",
            "YouTube reports transcripts are disabled.",
        )

        st.error(
            "Transcripts are disabled for this video."
        )

        return None

    except Exception:

        log_exception(
            "Failed to list available transcripts.",
            video_logger,
        )

        return None

    # -----------------------------------------------------
    # Prefer English
    # -----------------------------------------------------

    selected = None

    try:

        selected = (
            transcript_list.find_transcript(
                [
                    "en",
                    "en-US",
                    "en-GB",
                ]
            )
        )

        log(
            "info",
            (
                "Selected English transcript: "
                f"{selected.language} "
                f"({selected.language_code})"
            ),
        )

    except NoTranscriptFound:

        log(
            "warning",
            "English transcript unavailable.",
        )

    except Exception:

        log_exception(
            "Error while searching for English transcript.",
            video_logger,
        )

    # -----------------------------------------------------
    # Fallback transcript
    # -----------------------------------------------------

    if selected is None:

        try:

            selected = next(
                iter(transcript_list)
            )

            log(
                "info",
                (
                    "Using fallback transcript: "
                    f"{selected.language} "
                    f"({selected.language_code})"
                ),
            )

        except StopIteration:

            log(
                "error",
                "Transcript list is empty.",
            )

            return None

        except Exception:

            log_exception(
                (
                    "Failed to select "
                    "fallback transcript."
                ),
                video_logger,
            )

            return None

    # -----------------------------------------------------
    # Fetch selected transcript
    # -----------------------------------------------------

    try:

        log(
            "info",
            "Fetching selected transcript.",
        )

        fetched = selected.fetch()

        text = transcript_to_text(
            fetched
        )

        if not text:

            log(
                "error",
                "Transcript was fetched but contains no text.",
            )

            return None

        preview = (
            text[:500]
            .replace("\n", " ")
        )

        log(
            "info",
            (
                "Transcript successfully fetched. "
                f"Characters={len(text):,}"
            ),
        )

        log(
            "debug",
            f"Transcript preview: {preview}",
        )

        return text

    except TranscriptsDisabled:

        log(
            "error",
            "Transcripts are disabled.",
        )

        st.error(
            "Transcripts are disabled for this video."
        )

    except NoTranscriptFound:

        log(
            "error",
            "No transcript found after selection.",
        )

    except VideoUnavailable:

        log(
            "error",
            "Video unavailable during transcript fetch.",
        )

    except CouldNotRetrieveTranscript as error:

        log(
            "error",
            (
                "Could not retrieve transcript: "
                f"{error}"
            ),
        )

        st.error(
            f"Could not retrieve transcript: {error}"
        )

    except Exception:

        log_exception(
            "Failed to fetch selected transcript.",
            video_logger,
        )

        st.error(
            "An unexpected error occurred while "
            "retrieving the transcript."
        )

    return None


# =========================================================
# SAVE JSON UTILITY
# =========================================================

def save_json(path, data):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# BUILD VECTOR STORE
# =========================================================

def create_vector_store(
    transcript_text,
    video_id,
    video_logger=None,
):

    log(
        "info",
        "==================================================",
    )

    log(
        "info",
        "STARTING RAG PIPELINE",
    )

    log(
        "info",
        "==================================================",
    )

    if video_logger:
        video_logger.info(
            "STARTING RAG PIPELINE"
        )

    if not transcript_text:
        raise ValueError(
            "Transcript is empty."
        )

    # -----------------------------------------------------
    # Directory
    # -----------------------------------------------------

    output_dir = (
        RAG_ROOT / video_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "info",
        f"RAG storage directory: {output_dir}",
    )

    # -----------------------------------------------------
    # Transcript
    # -----------------------------------------------------

    transcript_path = (
        output_dir /
        TRANSCRIPT_FILENAME
    )

    with open(
        transcript_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            transcript_text
        )

    log(
        "info",
        f"Transcript saved: {transcript_path}",
    )

    # -----------------------------------------------------
    # Text loader
    # -----------------------------------------------------

    loader = TextLoader(
        str(transcript_path),
        encoding="utf-8",
    )

    documents = loader.load()

    log(
        "info",
        f"Loaded documents: {len(documents)}",
    )

    # -----------------------------------------------------
    # Recursive chunking
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    log(
        "info",
        (
            f"Created {len(chunks)} chunks "
            f"(chunk_size={CHUNK_SIZE}, "
            f"overlap={CHUNK_OVERLAP})"
        ),
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    # -----------------------------------------------------
    # Chunk metadata
    # -----------------------------------------------------

    chunks_data = []

    for index, document in enumerate(
        chunks
    ):

        text = (
            document.page_content
            .strip()
        )

        chunks_data.append(
            {
                "chunk_id": index,
                "text": text,
                "character_count": len(text),
                "word_count": len(
                    text.split()
                ),
            }
        )

    chunks_path = (
        output_dir /
        CHUNKS_FILENAME
    )

    save_json(
        chunks_path,
        chunks_data,
    )

    log(
        "info",
        f"Chunk data saved: {chunks_path}",
    )

    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------

    log(
        "info",
        (
            "Loading embedding model: "
            f"{EMBEDDING_MODEL}"
        ),
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    log(
        "info",
        "Embedding model loaded successfully.",
    )

    # -----------------------------------------------------
    # Generate vectors / FAISS
    # -----------------------------------------------------

    log(
        "info",
        "Generating embeddings and creating FAISS index.",
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    log(
        "info",
        "FAISS vector store created successfully.",
    )

    # -----------------------------------------------------
    # Embedding statistics
    # -----------------------------------------------------

    vector_count = (
        vector_store.index.ntotal
    )

    vector_dimension = (
        vector_store.index.d
    )

    log(
        "info",
        f"FAISS vector count: {vector_count}",
    )

    log(
        "info",
        f"Embedding dimension: {vector_dimension}",
    )

    # -----------------------------------------------------
    # Extract actual vectors
    # -----------------------------------------------------

    vectors = np.zeros(
        (
            vector_count,
            vector_dimension,
        ),
        dtype=np.float32,
    )

    if vector_count > 0:

        vector_store.index.reconstruct_n(
            0,
            vector_count,
            vectors,
        )

    embeddings_path = (
        output_dir /
        EMBEDDINGS_FILENAME
    )

    np.save(
        embeddings_path,
        vectors,
    )

    log(
        "info",
        f"Embedding matrix saved: {embeddings_path}",
    )

    log(
        "info",
        f"Embedding matrix shape: {vectors.shape}",
    )

    log(
        "info",
        (
            f"Embedding matrix size: "
            f"{vectors.nbytes / (1024 * 1024):.2f} MB"
        ),
    )

    # -----------------------------------------------------
    # Save FAISS
    # -----------------------------------------------------

    vector_store.save_local(
        str(output_dir)
    )

    log(
        "info",
        (
            "FAISS files saved in: "
            f"{output_dir}"
        ),
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    total_characters = sum(
        chunk["character_count"]
        for chunk in chunks_data
    )

    total_words = sum(
        chunk["word_count"]
        for chunk in chunks_data
    )

    average_chunk_length = (
        total_characters /
        len(chunks_data)
    )

    average_chunk_words = (
        total_words /
        len(chunks_data)
    )

    largest_chunk = max(
        chunk["character_count"]
        for chunk in chunks_data
    )

    smallest_chunk = min(
        chunk["character_count"]
        for chunk in chunks_data
    )

    metadata = {
        "video_id": video_id,
        "generated_at": datetime.now().isoformat(),

        "transcript": {
            "characters": len(
                transcript_text
            ),
            "words": len(
                transcript_text.split()
            ),
        },

        "chunking": {
            "chunk_count": len(
                chunks_data
            ),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "average_chunk_characters": round(
                average_chunk_length,
                2,
            ),
            "average_chunk_words": round(
                average_chunk_words,
                2,
            ),
            "smallest_chunk_characters": (
                smallest_chunk
            ),
            "largest_chunk_characters": (
                largest_chunk
            ),
        },

        "embeddings": {
            "model": EMBEDDING_MODEL,
            "count": vector_count,
            "dimension": vector_dimension,
            "dtype": str(
                vectors.dtype
            ),
        },

        "retrieval": {
            "top_k": TOP_K,
        },

        "gemini": {
            "model": GEMINI_MODEL,
        },
    }

    metadata_path = (
        output_dir /
        METADATA_FILENAME
    )

    save_json(
        metadata_path,
        metadata,
    )

    log(
        "info",
        f"Metadata saved: {metadata_path}",
    )

    # -----------------------------------------------------
    # Write processing event
    # -----------------------------------------------------

    if video_logger:

        video_logger.info(
            (
                "RAG COMPLETE | "
                f"transcript_chars={len(transcript_text)} | "
                f"chunks={len(chunks_data)} | "
                f"embeddings={vector_count} | "
                f"dimension={vector_dimension}"
            )
        )

    log(
        "info",
        "RAG storage completed successfully.",
    )

    log(
        "info",
        f"  Transcript : {transcript_path}",
    )

    log(
        "info",
        f"  Chunks     : {chunks_path}",
    )

    log(
        "info",
        f"  Embeddings : {embeddings_path}",
    )

    log(
        "info",
        f"  FAISS      : {output_dir}",
    )

    log(
        "info",
        f"  Metadata   : {metadata_path}",
    )

    return vector_store


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(
    vector_store,
    question,
):

    log(
        "info",
        (
            "Retrieving relevant context for: "
            f"{question}"
        ),
    )

    documents = vector_store.similarity_search(
        question,
        k=TOP_K,
    )

    log(
        "info",
        (
            f"Retrieved {len(documents)} "
            "relevant chunks."
        ),
    )

    context_parts = []

    retrieved_metadata = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        content = (
            document.page_content
            .strip()
        )

        log(
            "debug",
            (
                f"Retrieved chunk {index}: "
                f"{len(content)} characters."
            ),
        )

        retrieved_metadata.append(
            {
                "rank": index,
                "characters": len(
                    content
                ),
                "words": len(
                    content.split()
                ),
            }
        )

        context_parts.append(
            f"""
--- CONTEXT CHUNK {index} ---
{content}
"""
        )

    context = (
        "\n".join(
            context_parts
        )
        .strip()
    )

    log(
        "info",
        (
            f"Final retrieved context: "
            f"{len(context):,} characters."
        ),
    )

    return (
        context,
        retrieved_metadata,
    )


# =========================================================
# COUNT GEMINI TOKENS
# =========================================================

def count_gemini_tokens(
    contents,
):

    try:

        response = (
            gemini_client.models.count_tokens(
                model=GEMINI_MODEL,
                contents=contents,
            )
        )

        total_tokens = getattr(
            response,
            "total_tokens",
            None,
        )

        if total_tokens is not None:

            log(
                "info",
                (
                    "Gemini token count: "
                    f"{total_tokens:,}"
                ),
            )

        return total_tokens

    except Exception as error:

        log(
            "warning",
            (
                "Could not count Gemini tokens: "
                f"{error}"
            ),
        )

        return None


# =========================================================
# SAVE TOKEN USAGE
# =========================================================

def save_token_usage(
    video_id,
    question,
    context,
    retrieved_metadata,
    gemini_input_tokens,
    response,
):

    output_dir = (
        RAG_ROOT / video_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    prompt_tokens = None
    output_tokens = None
    total_tokens = None

    if response_usage:

        if isinstance(
            response_usage,
            dict,
        ):

            prompt_tokens = (
                response_usage.get(
                    "prompt_token_count"
                )
                or response_usage.get(
                    "promptTokenCount"
                )
                or response_usage.get(
                    "input_tokens"
                )
            )

            output_tokens = (
                response_usage.get(
                    "candidates_token_count"
                )
                or response_usage.get(
                    "candidatesTokenCount"
                )
                or response_usage.get(
                    "output_tokens"
                )
            )

            total_tokens = (
                response_usage.get(
                    "total_token_count"
                )
                or response_usage.get(
                    "totalTokenCount"
                )
                or response_usage.get(
                    "total_tokens"
                )
            )

        else:

            prompt_tokens = getattr(
                response_usage,
                "prompt_token_count",
                None,
            )

            output_tokens = getattr(
                response_usage,
                "candidates_token_count",
                None,
            )

            total_tokens = getattr(
                response_usage,
                "total_token_count",
                None,
            )

    usage_record = {
        "timestamp": datetime.now().isoformat(),

        "model": GEMINI_MODEL,

        "question": question,

        "question_characters": len(
            question
        ),

        "retrieved_context": {
            "characters": len(
                context
            ),
            "words": len(
                context.split()
            ),
            "chunks": retrieved_metadata,
        },

        "gemini": {
            "count_tokens_result": (
                gemini_input_tokens
            ),
            "prompt_tokens": (
                prompt_tokens
            ),
            "output_tokens": (
                output_tokens
            ),
            "total_tokens": (
                total_tokens
            ),
        },

        "answer": {
            "characters": len(
                response.text or ""
            ),
        },
    }

    usage_path = (
        output_dir /
        TOKEN_USAGE_FILENAME
    )

    # Preserve previous questions instead of overwriting.
    existing_records = []

    if usage_path.exists():

        try:

            with open(
                usage_path,
                "r",
                encoding="utf-8",
            ) as file:

                existing = json.load(
                    file
                )

                if isinstance(
                    existing,
                    list,
                ):

                    existing_records = existing

        except Exception:

            existing_records = []

    existing_records.append(
        usage_record
    )

    save_json(
        usage_path,
        existing_records,
    )

    log(
        "info",
        (
            f"Token usage saved: "
            f"{usage_path}"
        ),
    )

    log(
        "info",
        (
            "Gemini usage | "
            f"prompt={prompt_tokens} | "
            f"output={output_tokens} | "
            f"total={total_tokens}"
        ),
    )


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(
    video_id,
    question,
    context,
    retrieved_metadata,
):

    log(
        "info",
        "Sending question to Gemini.",
    )

    system_instruction = """
You are a YouTube video question-answering assistant.

Answer the user's question using the provided transcript context.

Rules:
1. Use the transcript context as the primary source.
2. Do not invent facts that are not supported by the transcript.
3. If the answer is not present in the retrieved context,
   clearly say that the transcript does not provide enough information.
4. You may answer in English even if the transcript is in another language.
5. Keep the answer clear, useful, and direct.
6. Preserve important technical terminology.
7. If the transcript is auto-generated and the wording seems imperfect,
   interpret it carefully using the surrounding context.
"""

    prompt = f"""
{system_instruction}

TRANSCRIPT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # -----------------------------------------------------
    # Count tokens BEFORE generation
    # -----------------------------------------------------

    log(
        "info",
        "Counting Gemini input tokens.",
    )

    input_token_count = (
        count_gemini_tokens(
            prompt
        )
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    try:

        response = (
            gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
        )

        answer = (
            response.text or ""
        ).strip()

        if not answer:

            raise ValueError(
                "Gemini returned an empty response."
            )

        log(
            "info",
            (
                "Gemini response received. "
                f"Characters={len(answer):,}"
            ),
        )

        log(
            "debug",
            f"Gemini answer: {answer}",
        )

        # -------------------------------------------------
        # Save usage
        # -------------------------------------------------

        save_token_usage(
            video_id=video_id,
            question=question,
            context=context,
            retrieved_metadata=retrieved_metadata,
            gemini_input_tokens=input_token_count,
            response=response,
        )

        return answer

    except Exception:

        log_exception(
            "Gemini request failed."
        )

        raise


# =========================================================
# MAIN UI
# =========================================================

st.title(
    "🎥 YouTube RAG Chatbot"
)

st.caption(
    (
        "YouTube Transcript → "
        "Chunking → Embeddings → FAISS → "
        "Gemini 3.6 Flash"
    )
)

st.write(
    (
        "Process a YouTube video's transcript, "
        "build a local vector index, and ask "
        "questions about the video."
    )
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write(
        f"**LLM:** `{GEMINI_MODEL}`"
    )

    st.write(
        f"**Embeddings:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Chunk size:** `{CHUNK_SIZE}`"
    )

    st.write(
        f"**Chunk overlap:** `{CHUNK_OVERLAP}`"
    )

    st.write(
        f"**Top-K:** `{TOP_K}`"
    )

    st.divider()

    st.write(
        "**Saved artifacts**"
    )

    st.code(
        (
            "transcript.txt\n"
            "chunks.json\n"
            "embeddings.npy\n"
            "index.faiss\n"
            "index.pkl\n"
            "metadata.json\n"
            "token_usage.json\n"
            "processing.log"
        ),
        language="text",
    )

    st.divider()

    if st.button(
        "Clear Logs"
    ):

        st.session_state[
            "logs"
        ] = []

        st.rerun()

    st.divider()

    st.caption(
        "Global log file:"
    )

    st.code(
        os.path.abspath(
            LOG_FILE
        ),
        language="text",
    )


# =========================================================
# VIDEO INPUT
# =========================================================

video_url = st.text_input(
    "YouTube URL",
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

    st.session_state[
        "logs"
    ] = []

    # Clear previous current-session vector store.
    st.session_state.pop(
        "vector_store",
        None,
    )

    st.session_state.pop(
        "video_id",
        None,
    )

    log(
        "info",
        "User clicked Process Video.",
    )

    if not video_url.strip():

        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        with st.spinner(
            (
                "Downloading transcript, "
                "creating embeddings, and "
                "building FAISS..."
            )
        ):

            try:

                # -----------------------------------------
                # Extract video ID
                # -----------------------------------------

                video_id = (
                    extract_video_id(
                        video_url
                    )
                )

                if not video_id:

                    st.error(
                        "Could not determine the YouTube video ID."
                    )

                    st.stop()

                # -----------------------------------------
                # Per-video logger
                # -----------------------------------------

                video_logger = (
                    create_video_logger(
                        video_id
                    )
                )

                video_logger.info(
                    f"Processing video: {video_id}"
                )

                # -----------------------------------------
                # Transcript
                # -----------------------------------------

                transcript = (
                    get_youtube_transcript(
                        video_url,
                        video_logger,
                    )
                )

                if transcript is None:

                    st.error(
                        "❌ Could not retrieve transcript."
                    )

                else:

                    transcript_length = (
                        len(transcript)
                    )

                    st.session_state[
                        "transcript_length"
                    ] = transcript_length

                    st.session_state[
                        "video_id"
                    ] = video_id

                    log(
                        "info",
                        (
                            "Transcript ready. "
                            f"Characters={transcript_length:,}"
                        ),
                    )

                    # ---------------------------------
                    # RAG
                    # ---------------------------------

                    vector_store = (
                        create_vector_store(
                            transcript_text=transcript,
                            video_id=video_id,
                            video_logger=video_logger,
                        )
                    )

                    st.session_state[
                        "vector_store"
                    ] = vector_store

                    st.session_state[
                        "video_processed"
                    ] = True

                    st.session_state[
                        "processing_directory"
                    ] = str(
                        RAG_ROOT /
                        video_id
                    )

                    st.success(
                        "✅ Video processed successfully."
                    )

                    st.info(
                        (
                            f"Transcript: "
                            f"{transcript_length:,} characters\n\n"
                            f"RAG data: "
                            f"`rag_data/{video_id}/`"
                        )
                    )

                    video_logger.info(
                        "VIDEO PROCESSING COMPLETED SUCCESSFULLY"
                    )

                    log(
                        "info",
                        (
                            "VIDEO PROCESSING COMPLETED "
                            "SUCCESSFULLY."
                        ),
                    )

            except Exception as error:

                log_exception(
                    (
                        "Video processing failed: "
                        f"{error}"
                    )
                )

                st.error(
                    f"❌ Processing failed: {error}"
                )


# =========================================================
# QUESTIONS
# =========================================================

if (
    "vector_store"
    in st.session_state
):

    st.divider()

    st.subheader(
        "💬 Ask About the Video"
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "What is LangSmith?"
        ),
    )

    if question.strip():

        video_id = st.session_state.get(
            "video_id"
        )

        if not video_id:

            st.error(
                "Video ID is missing from session."
            )

        else:

            log(
                "info",
                f"User question: {question}",
            )

            video_logger = (
                create_video_logger(
                    video_id
                )
            )

            with st.spinner(
                (
                    "Searching transcript "
                    "and generating answer..."
                )
            ):

                try:

                    # ---------------------------------
                    # Retrieval
                    # ---------------------------------

                    (
                        context,
                        retrieved_metadata,
                    ) = retrieve_context(
                        st.session_state[
                            "vector_store"
                        ],
                        question,
                    )

                    video_logger.info(
                        (
                            "QUESTION | "
                            f"{question}"
                        )
                    )

                    video_logger.info(
                        (
                            "RETRIEVAL | "
                            f"chunks={len(retrieved_metadata)} | "
                            f"context_chars={len(context)}"
                        )
                    )

                    # ---------------------------------
                    # Gemini
                    # ---------------------------------

                    answer = ask_gemini(
                        video_id=video_id,
                        question=question,
                        context=context,
                        retrieved_metadata=(
                            retrieved_metadata
                        ),
                    )

                    # ---------------------------------
                    # Display answer
                    # ---------------------------------

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(
                        answer
                    )

                    # ---------------------------------
                    # Retrieved context
                    # ---------------------------------

                    with st.expander(
                        "🔎 View Retrieved Transcript Context"
                    ):

                        st.text(
                            context
                        )

                    # ---------------------------------
                    # Token info
                    # ---------------------------------

                    usage_path = (
                        RAG_ROOT /
                        video_id /
                        TOKEN_USAGE_FILENAME
                    )

                    if usage_path.exists():

                        try:

                            with open(
                                usage_path,
                                "r",
                                encoding="utf-8",
                            ) as file:

                                usage_records = (
                                    json.load(file)
                                )

                            if usage_records:

                                latest_usage = (
                                    usage_records[-1]
                                )

                                gemini_usage = (
                                    latest_usage.get(
                                        "gemini",
                                        {}
                                    )
                                )

                                with st.expander(
                                    "📊 Token / Usage Information"
                                ):

                                    col1, col2, col3 = (
                                        st.columns(3)
                                    )

                                    with col1:

                                        st.metric(
                                            "Input Tokens",
                                            str(
                                                gemini_usage.get(
                                                    "prompt_tokens"
                                                )
                                                or
                                                gemini_usage.get(
                                                    "count_tokens_result"
                                                )
                                                or
                                                "N/A"
                                            ),
                                        )

                                    with col2:

                                        st.metric(
                                            "Output Tokens",
                                            str(
                                                gemini_usage.get(
                                                    "output_tokens"
                                                )
                                                or
                                                "N/A"
                                            ),
                                        )

                                    with col3:

                                        st.metric(
                                            "Total Tokens",
                                            str(
                                                gemini_usage.get(
                                                    "total_tokens"
                                                )
                                                or
                                                "N/A"
                                            ),
                                        )

                        except Exception as error:

                            log(
                                "warning",
                                (
                                    "Could not display "
                                    f"token usage: {error}"
                                ),
                            )

                except Exception as error:

                    log_exception(
                        (
                            "Failed to generate answer: "
                            f"{error}"
                        )
                    )

                    st.error(
                        (
                            "❌ Failed to generate answer: "
                            f"{error}"
                        )
                    )


# =========================================================
# LOG DISPLAY
# =========================================================

st.divider()

st.subheader(
    "🧾 Detailed Logs"
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
        "No logs yet."
    )


# =========================================================
# CURRENT RAG DIRECTORY
# =========================================================

if "processing_directory" in st.session_state:

    st.divider()

    st.subheader(
        "📁 Generated RAG Data"
    )

    directory = Path(
        st.session_state[
            "processing_directory"
        ]
    )

    st.code(
        str(
            directory.resolve()
        ),
        language="text",
    )

    if directory.exists():

        files = []

        for file in sorted(
            directory.iterdir()
        ):

            if file.is_file():

                size_mb = (
                    file.stat().st_size
                    / (1024 * 1024)
                )

                files.append(
                    (
                        file.name,
                        f"{size_mb:.2f} MB"
                    )
                )

        for filename, size in files:

            st.write(
                f"**{filename}** — {size}"
            )
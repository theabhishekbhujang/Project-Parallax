"""
🔮 Project Parallax - Advanced Document Intelligence Platform
A comprehensive single-file implementation with enterprise-grade features.

Author: AI Assistant
Version: 2.0.0 - Enterprise Edition
License: MIT
"""

import streamlit as st
import asyncio
import aiofiles
import os
import logging
import sqlite3
import json
import hashlib
import pickle
import time
import uuid
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create dummy objects to prevent errors
    class DummyPlotly:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    px = go = DummyPlotly()
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import base64
import io
import sys
from collections import defaultdict, Counter
import re
import math

# Core ML/AI imports
try:
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from langchain_ollama.llms import OllamaLLM
except ImportError as e:
    st.error(f"Missing required dependencies: {e}")
    st.stop()

# Advanced processing imports
try:
    import cv2
    import pytesseract
    from PIL import Image
    import camelot
    import spacy
    from transformers import pipeline
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from textstat import flesch_reading_ease
    import yake
    ADVANCED_FEATURES = True
except ImportError as e:
    ADVANCED_FEATURES = False
    # Create dummy modules to prevent errors
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
        
        def __call__(self, *args, **kwargs):
            return None
    
    # Assign dummy modules
    cv2 = pytesseract = Image = camelot = spacy = pipeline = nx = TfidfVectorizer = KMeans = flesch_reading_ease = yake = DummyModule()
    
    # Only show warning if streamlit is available (to avoid errors during import)
    try:
        import streamlit as st
        st.warning("⚠️ Some advanced features disabled. Install optional dependencies for full functionality.")
    except ImportError:
        pass  # Streamlit not available yet


# ================== CONFIGURATION SYSTEM ==================

@dataclass
class AdvancedConfig:
    """Comprehensive configuration system with environment support"""
    
    # Core Settings
    APP_TITLE: str = "🔮 Project Parallax - Enterprise"
    APP_SUBTITLE: str = "Advanced Document Intelligence Platform"
    VERSION: str = "2.0.0"
    
    # Storage & Database
    PDF_STORAGE_PATH: Path = Path(os.getenv('PDF_STORAGE_PATH', 'data/documents/'))
    DATABASE_PATH: Path = Path(os.getenv('DATABASE_PATH', 'data/parallax.db'))
    CACHE_PATH: Path = Path(os.getenv('CACHE_PATH', 'data/cache/'))
    
    # AI Models
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'deepseek-r1:8b')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'deepseek-r1:8b')
    LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE', '0.2'))
    
    # Processing Parameters
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '8000'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '500'))
    SIMILARITY_SEARCH_K: int = int(os.getenv('SIMILARITY_SEARCH_K', '5'))
    MAX_CONCURRENT_PROCESSES: int = int(os.getenv('MAX_CONCURRENT_PROCESSES', '4'))
    
    # Performance & Limits
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '50')) * 1024 * 1024
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    STREAM_DELAY: float = float(os.getenv('STREAM_DELAY', '0.02'))
    
    # Security & Privacy
    ENABLE_ENCRYPTION: bool = os.getenv('ENABLE_ENCRYPTION', 'False').lower() == 'true'
    AUDIT_LOGGING: bool = os.getenv('AUDIT_LOGGING', 'True').lower() == 'true'
    SESSION_TIMEOUT: int = int(os.getenv('SESSION_TIMEOUT', '1800'))
    
    # Analytics & Monitoring
    ENABLE_ANALYTICS: bool = os.getenv('ENABLE_ANALYTICS', 'True').lower() == 'true'
    ENABLE_REAL_TIME_METRICS: bool = os.getenv('ENABLE_REAL_TIME_METRICS', 'True').lower() == 'true'
    
    # Advanced Features
    ENABLE_MULTIMODAL: bool = ADVANCED_FEATURES and os.getenv('ENABLE_MULTIMODAL', 'True').lower() == 'true'
    ENABLE_NLP_ANALYSIS: bool = ADVANCED_FEATURES and os.getenv('ENABLE_NLP_ANALYSIS', 'True').lower() == 'true'
    ENABLE_VOICE_INTERFACE: bool = os.getenv('ENABLE_VOICE_INTERFACE', 'False').lower() == 'true'
    
    def __post_init__(self):
        """Initialize directories and validate configuration"""
        for path in [self.PDF_STORAGE_PATH, self.CACHE_PATH]:
            path.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ================== ADVANCED LOGGING SYSTEM ==================

class EnterpriseLogger:
    """Advanced logging system with structured logging and audit trails"""
    
    def __init__(self, name: str, config: AdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(name)
        self.audit_queue = queue.Queue()
        self._setup_logging()
        self._start_audit_processor()
    
    def _setup_logging(self):
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler for persistent logging
            if not (Path('logs').exists()):
                Path('logs').mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler('logs/parallax.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.setLevel(logging.INFO)
    
    def _start_audit_processor(self):
        """Start background audit log processor"""
        if self.config.AUDIT_LOGGING:
            def process_audit_logs():
                while True:
                    try:
                        audit_entry = self.audit_queue.get(timeout=1)
                        self._write_audit_log(audit_entry)
                        self.audit_queue.task_done()
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Audit logging error: {e}")
            
            audit_thread = threading.Thread(target=process_audit_logs, daemon=True)
            audit_thread.start()
    
    def _write_audit_log(self, entry: Dict[str, Any]):
        """Write audit entry to database"""
        try:
            with DatabaseManager().get_connection() as conn:
                conn.execute("""
                    INSERT INTO audit_logs (timestamp, user_id, action, resource, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entry['timestamp'],
                    entry.get('user_id', 'anonymous'),
                    entry['action'],
                    entry.get('resource', ''),
                    json.dumps(entry.get('details', {}))
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def audit(self, action: str, user_id: Optional[str] = None, resource: Optional[str] = None, **details):
        """Log audit event"""
        if self.config.AUDIT_LOGGING:
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id or 'anonymous',
                'action': action,
                'resource': resource,
                'details': details
            }
            self.audit_queue.put(audit_entry)
    
    def info(self, message: str, **kwargs):
        self.logger.info(f"{message} {kwargs if kwargs else ''}")
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        error_msg = f"{message} {kwargs if kwargs else ''}"
        if exception:
            error_msg += f" - Exception: {str(exception)}"
        self.logger.error(error_msg, exc_info=exception is not None)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(f"{message} {kwargs if kwargs else ''}")


# ================== DATABASE MANAGEMENT SYSTEM ==================

class DatabaseManager:
    """Advanced SQLite database manager with connection pooling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config = AdvancedConfig()
        self.logger = EnterpriseLogger(__name__, self.config)
        self.connection_pool = queue.Queue(maxsize=10)
        self._create_tables()
        
        # Pre-populate connection pool
        for _ in range(5):
            self.connection_pool.put(self._create_connection())
    
    def _create_connection(self):
        """Create a new database connection"""
        conn = sqlite3.connect(str(self.config.DATABASE_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            conn = self.connection_pool.get_nowait()
        except queue.Empty:
            conn = self._create_connection()
        
        try:
            yield conn
        finally:
            if self.connection_pool.qsize() < 10:
                self.connection_pool.put(conn)
            else:
                conn.close()
    
    def _create_tables(self):
        """Create database schema"""
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            file_size INTEGER,
            page_count INTEGER,
            processing_status TEXT DEFAULT 'pending',
            metadata TEXT,  -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER,
            start_char INTEGER,
            end_char INTEGER,
            metadata TEXT,  -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        );
        
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'anonymous',
            document_id TEXT,
            query TEXT NOT NULL,
            response TEXT,
            context_chunks TEXT,  -- JSON array of chunk IDs
            confidence_score REAL,
            processing_time REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        );
        
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'anonymous',
            ip_address TEXT,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        
        CREATE TABLE IF NOT EXISTS analytics_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            user_id TEXT DEFAULT 'anonymous',
            session_id TEXT,
            document_id TEXT,
            event_data TEXT,  -- JSON
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            user_id TEXT,
            action TEXT NOT NULL,
            resource TEXT,
            details TEXT  -- JSON
        );
        
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata TEXT,  -- JSON
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_document ON conversations(document_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type);
        """
        
        with self.get_connection() as conn:
            conn.executescript(schema)
            conn.commit()


# ================== ADVANCED CACHING SYSTEM ==================

class IntelligentCache:
    """Advanced caching system with TTL and intelligent invalidation"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.cache_dir = config.CACHE_PATH
        self.memory_cache = {}
        self.cache_metadata = {}
        self.lock = threading.Lock()
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a safe cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        cache_key = self._get_cache_key(key)
        
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                metadata = self.cache_metadata.get(cache_key, {})
                if self._is_valid(metadata):
                    return self.memory_cache[cache_key]
                else:
                    self._evict_from_memory(cache_key)
            
            # Check disk cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        metadata = data.get('metadata', {})
                        
                        if self._is_valid(metadata):
                            # Load back into memory cache
                            self.memory_cache[cache_key] = data['value']
                            self.cache_metadata[cache_key] = metadata
                            return data['value']
                        else:
                            cache_path.unlink()  # Remove expired cache
                except Exception:
                    pass  # Corrupted cache, ignore
            
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        cache_key = self._get_cache_key(key)
        ttl = ttl or self.config.CACHE_TTL
        
        metadata = {
            'created_at': time.time(),
            'ttl': ttl,
            'expires_at': time.time() + ttl
        }
        
        with self.lock:
            # Store in memory
            self.memory_cache[cache_key] = value
            self.cache_metadata[cache_key] = metadata
            
            # Store on disk for persistence
            try:
                cache_path = self._get_cache_path(cache_key)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'metadata': metadata
                    }, f)
            except Exception as e:
                pass  # Non-critical, memory cache still works
    
    def _is_valid(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if not metadata:
            return False
        return time.time() < metadata.get('expires_at', 0)
    
    def _evict_from_memory(self, cache_key: str) -> None:
        """Remove item from memory cache"""
        self.memory_cache.pop(cache_key, None)
        self.cache_metadata.pop(cache_key, None)
    
    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.memory_cache.clear()
            self.cache_metadata.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass


# ================== ADVANCED DOCUMENT PROCESSING ==================

class AdvancedDocumentProcessor:
    """Comprehensive document processing with multi-modal capabilities"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.logger = EnterpriseLogger(__name__, config)
        self.cache = IntelligentCache(config)
        self.db = DatabaseManager()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_PROCESSES)
        
        self._initialize_ai_models()
        self._initialize_nlp_models()
    
    def _initialize_ai_models(self):
        """Initialize AI models with error handling"""
        try:
            self.embedding_model = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL)
            self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)
            self.llm = OllamaLLM(
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            self.logger.info("AI models initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize AI models", exception=e)
            raise
    
    def _initialize_nlp_models(self):
        """Initialize advanced NLP models if available"""
        self.nlp_models = {}
        
        if ADVANCED_FEATURES and self.config.ENABLE_NLP_ANALYSIS:
            try:
                # Try to load spaCy model
                try:
                    self.nlp_models['spacy'] = spacy.load("en_core_web_sm")
                except OSError:
                    self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                
                # Initialize transformers pipelines
                self.nlp_models['sentiment'] = pipeline("sentiment-analysis", return_all_scores=True)
                self.nlp_models['summarization'] = pipeline("summarization", model="facebook/bart-large-cnn")
                self.nlp_models['ner'] = pipeline("ner", aggregation_strategy="simple")
                
                # Initialize keyword extraction
                self.nlp_models['keyword_extractor'] = yake.KeywordExtractor()
                
                self.logger.info("Advanced NLP models initialized")
            except Exception as e:
                self.logger.warning("Some NLP models failed to initialize", exception=e)
    
    async def process_document_comprehensive(self, file_path: Path, document_id: str) -> Dict[str, Any]:
        """Comprehensive document processing pipeline"""
        try:
            start_time = time.time()
            
            # 1. Basic document loading and chunking
            basic_analysis = await self._basic_document_analysis(file_path, document_id)
            
            # 2. Advanced content analysis
            advanced_analysis = await self._advanced_content_analysis(basic_analysis['chunks'])
            
            # 3. Multi-modal analysis (if enabled)
            multimodal_analysis = {}
            if self.config.ENABLE_MULTIMODAL:
                multimodal_analysis = await self._multimodal_analysis(file_path)
            
            # 4. Vector indexing
            await self._index_document_chunks(basic_analysis['chunks'])
            
            # 5. Generate document insights
            insights = await self._generate_document_insights(
                basic_analysis['chunks'],
                advanced_analysis,
                multimodal_analysis
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'document_id': document_id,
                'basic_analysis': basic_analysis,
                'advanced_analysis': advanced_analysis,
                'multimodal_analysis': multimodal_analysis,
                'insights': insights,
                'processing_time': processing_time,
                'status': 'completed'
            }
            
            # Store results in database
            await self._store_processing_results(document_id, result)
            
            self.logger.info("Document processing completed", 
                           document_id=document_id, 
                           processing_time=processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error("Document processing failed", 
                            document_id=document_id, 
                            exception=e)
            raise
    
    async def _basic_document_analysis(self, file_path: Path, document_id: str) -> Dict[str, Any]:
        """Basic document loading and analysis"""
        # Load PDF
        loader = PDFPlumberLoader(str(file_path))
        documents = await asyncio.to_thread(loader.load)
        
        if not documents:
            raise ValueError("No content extracted from PDF")
        
        # Create chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Calculate basic statistics
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Calculate readability
        readability_scores = {}
        if ADVANCED_FEATURES:
            full_text = " ".join(doc.page_content for doc in documents)
            readability_scores = {
                'flesch_reading_ease': flesch_reading_ease(full_text),
                'estimated_reading_time': total_words / 200  # 200 WPM average
            }
        
        return {
            'documents': documents,
            'chunks': chunks,
            'page_count': len(documents),
            'chunk_count': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'readability': readability_scores
        }
    
    async def _advanced_content_analysis(self, chunks: List[Document]) -> Dict[str, Any]:
        """Advanced NLP analysis of content"""
        if not self.config.ENABLE_NLP_ANALYSIS or not ADVANCED_FEATURES:
            return {}
        
        full_text = " ".join(chunk.page_content for chunk in chunks)
        
        analysis_results = {}
        
        try:
            # Named Entity Recognition
            if 'ner' in self.nlp_models:
                entities = await asyncio.to_thread(self.nlp_models['ner'], full_text[:1000])
                analysis_results['entities'] = entities
            
            # Sentiment Analysis
            if 'sentiment' in self.nlp_models:
                # Analyze sentiment of chunks
                chunk_sentiments = []
                for chunk in chunks[:10]:  # Limit for performance
                    sentiment = await asyncio.to_thread(
                        self.nlp_models['sentiment'], 
                        chunk.page_content[:500]
                    )
                    chunk_sentiments.append(sentiment[0])
                analysis_results['sentiment_analysis'] = chunk_sentiments
            
            # Keyword Extraction
            if 'keyword_extractor' in self.nlp_models:
                keywords = self.nlp_models['keyword_extractor'].extract_keywords(full_text[:2000])
                analysis_results['keywords'] = [(kw[1], kw[0]) for kw in keywords[:20]]
            
            # Topic Modeling (simple TF-IDF clustering)
            if len(chunks) > 5:
                topics = await self._extract_topics(chunks)
                analysis_results['topics'] = topics
            
        except Exception as e:
            self.logger.warning("Advanced content analysis partially failed", exception=e)
        
        return analysis_results
    
    async def _extract_topics(self, chunks: List[Document], n_topics: int = 5) -> List[Dict[str, Any]]:
        """Extract topics using TF-IDF and clustering"""
        try:
            # Check if sklearn is available
            if not ADVANCED_FEATURES:
                return []
            
            # Prepare text data
            texts = [chunk.page_content for chunk in chunks]
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = await asyncio.to_thread(vectorizer.fit_transform, texts)
            
            # K-means clustering
            n_clusters = min(n_topics, len(chunks) // 2)
            if n_clusters < 2:
                return []
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = await asyncio.to_thread(kmeans.fit_predict, tfidf_matrix)
            
            # Extract top terms for each cluster
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for i in range(n_clusters):
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Count documents in this cluster
                doc_count = sum(1 for label in cluster_labels if label == i)
                
                topics.append({
                    'topic_id': i,
                    'top_terms': top_terms,
                    'document_count': doc_count,
                    'strength': float(cluster_center.max())
                })
            
            return topics
            
        except Exception as e:
            self.logger.warning("Topic extraction failed", exception=e)
            return []
    
    async def _multimodal_analysis(self, file_path: Path) -> Dict[str, Any]:
        """Multi-modal analysis: images, tables, charts"""
        if not self.config.ENABLE_MULTIMODAL:
            return {}
        
        analysis = {
            'images': [],
            'tables': [],
            'charts': []
        }
        
        try:
            # Extract tables using camelot (if available)
            if ADVANCED_FEATURES:
                tables = camelot.read_pdf(str(file_path), pages='all', flavor='lattice')
                if tables is not None:
                    for i, table in enumerate(tables):
                        if table.accuracy > 50:  # Only include high-confidence tables
                            analysis['tables'].append({
                                'table_id': i,
                                'page': table.page,
                                'accuracy': table.accuracy,
                                'shape': table.df.shape,
                                'preview': table.df.head().to_dict('records')
                            })
            
        except Exception as e:
            self.logger.warning("Table extraction failed", exception=e)
        
        return analysis
    
    async def _index_document_chunks(self, chunks: List[Document]) -> None:
        """Index document chunks in vector store"""
        await asyncio.to_thread(self.vector_store.add_documents, chunks)
    
    async def _generate_document_insights(self, chunks: List[Document], 
                                        advanced_analysis: Dict[str, Any],
                                        multimodal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive document insights"""
        insights = {
            'summary': await self._generate_smart_summary(chunks),
            'key_metrics': self._calculate_key_metrics(chunks, advanced_analysis),
            'content_structure': self._analyze_content_structure(chunks),
            'complexity_analysis': self._analyze_complexity(chunks, advanced_analysis)
        }
        
        return insights
    
    async def _generate_smart_summary(self, chunks: List[Document]) -> Dict[str, str]:
        """Generate intelligent document summary"""
        try:
            # Use first few chunks for summary
            content_for_summary = " ".join(
                chunk.page_content for chunk in chunks[:3]
            )[:2000]  # Limit content length
            
            # Generate extractive summary
            sentences = content_for_summary.split('. ')
            if len(sentences) > 3:
                # Simple extractive summary - take first, middle, and last sentences
                extractive_summary = '. '.join([
                    sentences[0],
                    sentences[len(sentences)//2],
                    sentences[-1]
                ])
            else:
                extractive_summary = content_for_summary
            
            # Generate AI summary using LLM
            ai_summary_prompt = f"""
            Provide a concise summary of the following document content in 2-3 sentences:
            
            {content_for_summary[:1000]}
            
            Summary:
            """
            
            ai_summary = ""
            try:
                async for chunk in self.llm.astream(ai_summary_prompt):
                    ai_summary += chunk
            except Exception as e:
                ai_summary = "AI summary unavailable"
                self.logger.warning("AI summary generation failed", exception=e)
            
            return {
                'extractive_summary': extractive_summary,
                'ai_summary': ai_summary.strip(),
                'length': 'short' if len(chunks) < 10 else 'medium' if len(chunks) < 50 else 'long'
            }
            
        except Exception as e:
            self.logger.warning("Summary generation failed", exception=e)
            return {
                'extractive_summary': 'Summary unavailable',
                'ai_summary': 'Summary unavailable',
                'length': 'unknown'
            }
    
    def _calculate_key_metrics(self, chunks: List[Document], 
                             advanced_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key document metrics"""
        total_words = sum(len(chunk.page_content.split()) for chunk in chunks)
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        
        metrics = {
            'word_count': total_words,
            'character_count': total_chars,
            'average_words_per_chunk': total_words / len(chunks) if chunks else 0,
            'estimated_reading_time_minutes': total_words / 200,  # 200 WPM
        }
        
        # Add advanced metrics if available
        if advanced_analysis:
            if 'keywords' in advanced_analysis:
                metrics['unique_keywords'] = len(advanced_analysis['keywords'])
            if 'entities' in advanced_analysis:
                metrics['named_entities'] = len(advanced_analysis['entities'])
            if 'topics' in advanced_analysis:
                metrics['identified_topics'] = len(advanced_analysis['topics'])
        
        return metrics
    
    def _analyze_content_structure(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze document structure and organization"""
        # Simple heuristics for document structure analysis
        headings = []
        paragraphs = 0
        lists = 0
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Count paragraphs (rough estimate)
            paragraphs += content.count('\n\n')
            
            # Count lists (rough estimate)
            lists += content.count('• ') + content.count('- ') + content.count('1. ')
            
            # Find potential headings (lines that are short and followed by longer content)
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if (len(line.strip()) < 50 and 
                    line.strip() and 
                    i < len(lines) - 1 and 
                    len(lines[i+1].strip()) > len(line.strip())):
                    headings.append(line.strip())
        
        return {
            'estimated_headings': len(set(headings)),  # Remove duplicates
            'estimated_paragraphs': paragraphs,
            'estimated_lists': lists,
            'structure_complexity': 'simple' if len(headings) < 5 else 'moderate' if len(headings) < 15 else 'complex'
        }
    
    def _analyze_complexity(self, chunks: List[Document], 
                          advanced_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document complexity"""
        # Calculate various complexity metrics
        total_text = " ".join(chunk.page_content for chunk in chunks)
        
        # Basic complexity metrics
        avg_sentence_length = 0
        sentence_count = 0
        
        for chunk in chunks:
            sentences = [s.strip() for s in chunk.page_content.split('.') if s.strip()]
            sentence_count += len(sentences)
            if sentences:
                avg_sentence_length += sum(len(s.split()) for s in sentences) / len(sentences)
        
        avg_sentence_length = avg_sentence_length / len(chunks) if chunks else 0
        
        complexity = {
            'average_sentence_length': avg_sentence_length,
            'total_sentences': sentence_count,
            'complexity_score': min(10, avg_sentence_length / 3),  # Rough complexity score (0-10)
        }
        
        # Add readability from advanced analysis if available
        if 'readability' in advanced_analysis:
            complexity.update(advanced_analysis['readability'])
        
        return complexity
    
    async def _store_processing_results(self, document_id: str, results: Dict[str, Any]) -> None:
        """Store processing results in database"""
        try:
            with self.db.get_connection() as conn:
                # Update document status
                conn.execute("""
                    UPDATE documents 
                    SET processing_status = 'completed',
                        metadata = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (json.dumps(results), document_id))
                
                # Store chunks
                for i, chunk in enumerate(results['basic_analysis']['chunks']):
                    chunk_id = str(uuid.uuid4())
                    conn.execute("""
                        INSERT INTO document_chunks 
                        (id, document_id, content, chunk_index, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        document_id,
                        chunk.page_content,
                        i,
                        json.dumps(chunk.metadata)
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error("Failed to store processing results", exception=e)
    
    async def intelligent_search(self, query: str, document_id: Optional[str] = None, 
                               search_type: str = "semantic") -> List[Dict[str, Any]]:
        """Advanced search with multiple search strategies"""
        try:
            start_time = time.time()
            
            if search_type == "semantic":
                results = await self._semantic_search(query, document_id)
            elif search_type == "keyword":
                results = await self._keyword_search(query, document_id)
            elif search_type == "hybrid":
                semantic_results = await self._semantic_search(query, document_id)
                keyword_results = await self._keyword_search(query, document_id)
                results = self._merge_search_results(semantic_results, keyword_results)
            else:
                results = await self._semantic_search(query, document_id)
            
            search_time = time.time() - start_time
            
            # Log search analytics
            await self._log_search_analytics(query, len(results), search_time, search_type)
            
            return results
            
        except Exception as e:
            self.logger.error("Search failed", query=query, exception=e)
            return []
    
    async def _semantic_search(self, query: str, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic vector search"""
        # Use cached results if available
        cache_key = f"semantic_search:{query}:{document_id}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            return cached_results
        
        # Perform semantic search
        similar_docs = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            k=self.config.SIMILARITY_SEARCH_K
        )
        
        results = []
        for doc, score in similar_docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score),
                'search_type': 'semantic'
            })
        
        # Cache results
        self.cache.set(cache_key, results, ttl=1800)  # 30 minutes
        
        return results
    
    async def _keyword_search(self, query: str, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Traditional keyword-based search"""
        cache_key = f"keyword_search:{query}:{document_id}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            return cached_results
        
        results = []
        
        try:
            with self.db.get_connection() as conn:
                # Search in document chunks
                search_query = f"%{query.lower()}%"
                cursor = conn.execute("""
                    SELECT dc.content, dc.metadata, d.title
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE LOWER(dc.content) LIKE ?
                    ORDER BY dc.chunk_index
                    LIMIT ?
                """, (search_query, self.config.SIMILARITY_SEARCH_K))
                
                for row in cursor.fetchall():
                    # Calculate simple relevance score based on keyword frequency
                    content = row['content']
                    keyword_count = content.lower().count(query.lower())
                    relevance_score = min(1.0, keyword_count / 10)  # Normalize to 0-1
                    
                    results.append({
                        'content': content,
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'document_title': row['title'],
                        'relevance_score': relevance_score,
                        'search_type': 'keyword'
                    })
        
        except Exception as e:
            self.logger.error("Keyword search failed", exception=e)
        
        # Cache results
        self.cache.set(cache_key, results, ttl=1800)
        
        return results
    
    def _merge_search_results(self, semantic_results: List[Dict], 
                            keyword_results: List[Dict]) -> List[Dict]:
        """Merge and rank results from different search methods"""
        all_results = []
        
        # Add semantic results with adjusted scores
        for result in semantic_results:
            result['combined_score'] = result['similarity_score'] * 0.7  # Weight semantic search
            all_results.append(result)
        
        # Add keyword results with adjusted scores
        for result in keyword_results:
            result['combined_score'] = result['relevance_score'] * 0.3  # Weight keyword search
            all_results.append(result)
        
        # Remove duplicates based on content similarity and sort by combined score
        unique_results = []
        seen_content = set()
        
        for result in sorted(all_results, key=lambda x: x['combined_score'], reverse=True):
            content_hash = hashlib.md5(result['content'][:200].encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results[:self.config.SIMILARITY_SEARCH_K]
    
    async def _log_search_analytics(self, query: str, result_count: int, 
                                  search_time: float, search_type: str) -> None:
        """Log search analytics for monitoring and improvement"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO analytics_events 
                    (id, event_type, event_data, timestamp)
                    VALUES (?, 'search_performed', ?, CURRENT_TIMESTAMP)
                """, (
                    str(uuid.uuid4()),
                    json.dumps({
                        'query': query,
                        'result_count': result_count,
                        'search_time': search_time,
                        'search_type': search_type
                    })
                ))
                conn.commit()
        except Exception as e:
            self.logger.warning("Failed to log search analytics", exception=e)
    
    async def generate_enhanced_response(self, query: str, context_docs: List[Dict[str, Any]], 
                                       conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate enhanced response with multiple capabilities"""
        try:
            start_time = time.time()
            
            # Prepare context
            context_text = "\n\n".join([
                f"[Score: {doc.get('similarity_score', doc.get('relevance_score', 0)):.2f}] {doc['content']}"
                for doc in context_docs
            ])
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(query, context_text)
            
            # Generate response
            response_text = ""
            confidence_indicators = []
            
            async for chunk in self.llm.astream(prompt):
                response_text += chunk
                
                # Simple confidence tracking based on response patterns
                if any(phrase in chunk.lower() for phrase in ["according to", "the document states", "based on"]):
                    confidence_indicators.append("high")
                elif any(phrase in chunk.lower() for phrase in ["might", "possibly", "unclear"]):
                    confidence_indicators.append("low")
            
            # Calculate response metrics
            response_time = time.time() - start_time
            confidence_score = self._calculate_confidence_score(
                response_text, context_docs, confidence_indicators
            )
            
            # Extract citations and generate follow-up questions
            citations = self._extract_citations(response_text, context_docs)
            follow_up_questions = await self._generate_follow_up_questions(query, response_text)
            
            # Store conversation in database
            await self._store_conversation(
                conversation_id or str(uuid.uuid4()),
                query,
                response_text,
                context_docs,
                confidence_score,
                response_time
            )
            
            return {
                'response': response_text,
                'confidence_score': confidence_score,
                'response_time': response_time,
                'citations': citations,
                'follow_up_questions': follow_up_questions,
                'context_used': len(context_docs),
                'metadata': {
                    'model': self.config.LLM_MODEL,
                    'temperature': self.config.LLM_TEMPERATURE,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error("Enhanced response generation failed", exception=e)
            return {
                'response': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'confidence_score': 0.0,
                'error': True
            }
    
    def _build_enhanced_prompt(self, query: str, context: str) -> str:
        """Build enhanced prompt with better instructions"""
        return f"""You are an advanced AI assistant specialized in document analysis and question answering.

Your task is to provide accurate, comprehensive, and well-structured answers based on the provided context.

Guidelines:
1. Use ONLY the information provided in the context
2. If the context doesn't contain sufficient information, clearly state this limitation
3. Structure your response with clear headings when appropriate
4. Cite specific parts of the context when making claims
5. Be precise and avoid speculation
6. If multiple perspectives exist in the context, present them fairly
7. Highlight key insights and important details
8. Use bullet points or numbered lists when helpful for clarity

Context Information:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above:"""
    
    def _calculate_confidence_score(self, response: str, context_docs: List[Dict], 
                                  confidence_indicators: List[str]) -> float:
        """Calculate confidence score for the response"""
        score = 0.5  # Base score
        
        # Adjust based on context quality
        if context_docs:
            avg_similarity = sum(
                doc.get('similarity_score', doc.get('relevance_score', 0.5))
                for doc in context_docs
            ) / len(context_docs)
            score += avg_similarity * 0.3
        
        # Adjust based on confidence indicators
        if confidence_indicators:
            high_confidence_ratio = confidence_indicators.count("high") / len(confidence_indicators)
            score += high_confidence_ratio * 0.2 - (1 - high_confidence_ratio) * 0.1
        
        # Adjust based on response length and structure
        if len(response) > 100 and any(marker in response for marker in [":", "-", "•", "1.", "2."]):
            score += 0.1  # Well-structured response
        
        return max(0.0, min(1.0, score))
    
    def _extract_citations(self, response: str, context_docs: List[Dict]) -> List[Dict[str, str]]:
        """Extract and format citations from context documents"""
        citations = []
        
        for i, doc in enumerate(context_docs):
            # Simple citation extraction - could be enhanced with better NLP
            doc_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            
            citations.append({
                'id': i + 1,
                'preview': doc_preview,
                'score': doc.get('similarity_score', doc.get('relevance_score', 0)),
                'source': doc.get('metadata', {}).get('source', 'Document')
            })
        
        return citations
    
    async def _generate_follow_up_questions(self, original_query: str, response: str) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            follow_up_prompt = f"""Based on this conversation:
            
Question: {original_query}
Answer: {response[:500]}...

Generate 3 relevant follow-up questions that a user might want to ask next. 
Make them specific and actionable. Format as a simple list:

1.
2.
3."""
            
            follow_up_text = ""
            async for chunk in self.llm.astream(follow_up_prompt):
                follow_up_text += chunk
            
            # Extract questions from response
            questions = []
            for line in follow_up_text.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.')) or line.startswith('-')):
                    question = re.sub(r'^[\d\.\-\s]+', '', line).strip()
                    if question and question.endswith('?'):
                        questions.append(question)
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            self.logger.warning("Follow-up question generation failed", exception=e)
            return []
    
    async def _store_conversation(self, conversation_id: str, query: str, response: str,
                                context_docs: List[Dict], confidence_score: float,
                                response_time: float) -> None:
        """Store conversation in database"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO conversations 
                    (id, query, response, context_chunks, confidence_score, processing_time, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    conversation_id,
                    query,
                    response,
                    json.dumps([doc.get('metadata', {}) for doc in context_docs]),
                    confidence_score,
                    response_time
                ))
                conn.commit()
        except Exception as e:
            self.logger.warning("Failed to store conversation", exception=e)


# ================== ANALYTICS & MONITORING ==================

class AnalyticsEngine:
    """Comprehensive analytics and monitoring system"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.logger = EnterpriseLogger(__name__, config)
        self.db = DatabaseManager()
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics"""
        try:
            with self.db.get_connection() as conn:
                # Document metrics
                doc_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as processed,
                        SUM(CASE WHEN processing_status = 'pending' THEN 1 ELSE 0 END) as pending,
                        AVG(page_count) as avg_pages,
                        SUM(file_size) as total_size
                    FROM documents
                """).fetchone()
                
                # Conversation metrics
                conv_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_conversations,
                        AVG(confidence_score) as avg_confidence,
                        AVG(processing_time) as avg_response_time,
                        COUNT(DISTINCT DATE(timestamp)) as active_days
                    FROM conversations
                    WHERE timestamp > datetime('now', '-30 days')
                """).fetchone()
                
                # Recent activity
                recent_activity = conn.execute("""
                    SELECT 
                        event_type,
                        COUNT(*) as count,
                        DATE(timestamp) as date
                    FROM analytics_events
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY event_type, DATE(timestamp)
                    ORDER BY timestamp DESC
                    LIMIT 20
                """).fetchall()
                
                # Search analytics
                search_stats = conn.execute("""
                    SELECT 
                        json_extract(event_data, '$.search_type') as search_type,
                        COUNT(*) as count,
                        AVG(CAST(json_extract(event_data, '$.search_time') AS REAL)) as avg_time,
                        AVG(CAST(json_extract(event_data, '$.result_count') AS REAL)) as avg_results
                    FROM analytics_events
                    WHERE event_type = 'search_performed'
                    AND timestamp > datetime('now', '-7 days')
                    GROUP BY json_extract(event_data, '$.search_type')
                """).fetchall()
                
                return {
                    'documents': dict(doc_stats) if doc_stats else {},
                    'conversations': dict(conv_stats) if conv_stats else {},
                    'recent_activity': [dict(row) for row in recent_activity],
                    'search_analytics': [dict(row) for row in search_stats],
                    'system_health': await self._get_system_health(),
                    'updated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("Failed to get dashboard metrics", exception=e)
            return {}
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application metrics
            cache_size = len(IntelligentCache(self.config).memory_cache)
            
            # Database metrics
            with self.db.get_connection() as conn:
                db_size = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()").fetchone()
                
            return {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'cache_entries': cache_size,
                'database_size_mb': dict(db_size)['size'] / (1024*1024) if db_size else 0,
                'status': 'healthy' if cpu_usage < 80 and memory.percent < 85 else 'warning'
            }
            
        except ImportError:
            # psutil not available
            return {'status': 'unknown', 'message': 'System monitoring not available'}
        except Exception as e:
            self.logger.warning("System health check failed", exception=e)
            return {'status': 'unknown'}
    
    def log_user_action(self, action: str, user_id: Optional[str] = None, **kwargs) -> None:
        """Log user action for analytics"""
        if not self.config.ENABLE_ANALYTICS:
            return
        
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO analytics_events (id, event_type, user_id, event_data, timestamp)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    str(uuid.uuid4()),
                    action,
                    user_id or 'anonymous',
                    json.dumps(kwargs)
                ))
                conn.commit()
        except Exception as e:
            self.logger.warning("Failed to log user action", exception=e)


# ================== ADVANCED UI COMPONENTS ==================

class ModernUI:
    """Advanced UI components and layouts"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.logger = EnterpriseLogger(__name__, config)
        self.analytics = AnalyticsEngine(config)
    
    def setup_page_config(self):
        """Setup advanced page configuration"""
        st.set_page_config(
            page_title=f"{self.config.APP_TITLE} v{self.config.VERSION}",
            page_icon="🔮",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/project-parallax',
                'Report a bug': 'https://github.com/yourusername/project-parallax/issues',
                'About': f"Project Parallax v{self.config.VERSION} - Advanced Document Intelligence Platform"
            }
        )
    
    def apply_modern_styling(self):
        """Apply advanced CSS styling"""
        st.markdown("""
        <style>
            /* Import modern fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Global styling */
            html, body, [class*="css"]  {
                font-family: 'Inter', sans-serif;
            }
            
            .stApp {
                background: linear-gradient(135deg, #0f1419 0%, #1a202c 50%, #2d3748 100%);
                color: #e2e8f0;
            }
            
            /* Header styling */
            .main-header {
                background: linear-gradient(90deg, #1a202c 0%, #2d3748 100%);
                padding: 2rem;
                border-radius: 20px;
                margin-bottom: 2rem;
                border: 1px solid #4a5568;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            
            .main-title {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(45deg, #00d4ff, #0ea5e9, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            
            .subtitle {
                font-size: 1.1rem;
                color: #94a3b8;
                text-align: center;
                font-weight: 400;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background-color: #1a202c;
                border-right: 2px solid #4a5568;
            }
            
            /* Metrics cards */
            .metric-card {
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                padding: 1.5rem;
                border-radius: 15px;
                border: 1px solid #718096;
                margin: 0.5rem 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                transition: transform 0.2s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
            }
            
            /* File uploader */
            .stFileUploader > div > div > div {
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                border: 3px dashed #00d4ff;
                border-radius: 20px;
                padding: 3rem;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .stFileUploader > div > div > div:hover {
                border-color: #0ea5e9;
                background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
                transform: scale(1.02);
            }
            
            /* Chat interface */
            .stChatMessage {
                background: rgba(45, 55, 72, 0.8);
                backdrop-filter: blur(10px);
                border: 1px solid #4a5568;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(45deg, #00d4ff, #0ea5e9);
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                padding: 0.5rem 1.5rem;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                background: linear-gradient(45deg, #0ea5e9, #06b6d4);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
            }
            
            /* Progress bars */
            .stProgress > div > div > div {
                background: linear-gradient(90deg, #00d4ff, #0ea5e9);
            }
            
            /* Selectboxes and inputs */
            .stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea > div > div > textarea {
                background-color: #2d3748;
                border: 2px solid #4a5568;
                border-radius: 10px;
                color: #e2e8f0;
            }
            
            .stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
                border-color: #00d4ff;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            }
            
            /* Tables */
            .stDataFrame {
                border: 1px solid #4a5568;
                border-radius: 10px;
                overflow: hidden;
            }
            
            /* Expandable sections */
            .streamlit-expanderHeader {
                background: linear-gradient(90deg, #2d3748, #4a5568);
                border-radius: 10px;
                border: 1px solid #718096;
            }
            
            /* Alerts */
            .stAlert {
                border-radius: 10px;
                border: none;
            }
            
            .stSuccess {
                background: linear-gradient(135deg, #065f46, #047857);
                color: #d1fae5;
            }
            
            .stError {
                background: linear-gradient(135deg, #7f1d1d, #991b1b);
                color: #fecaca;
            }
            
            .stWarning {
                background: linear-gradient(135deg, #78350f, #92400e);
                color: #fde68a;
            }
            
            .stInfo {
                background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
                color: #dbeafe;
            }
            
            /* Loading animations */
            .stSpinner > div {
                border-top-color: #00d4ff !important;
                border-right-color: #00d4ff !important;
            }
            
            /* Custom scrollbars */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #2d3748;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #00d4ff, #0ea5e9);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #0ea5e9, #06b6d4);
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render modern application header"""
        st.markdown(f"""
        <div class="main-header">
            <h1 class="main-title">{self.config.APP_TITLE}</h1>
            <p class="subtitle">{self.config.APP_SUBTITLE}</p>
            <div style="text-align: center; margin-top: 1rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Version {self.config.VERSION} | 
                Powered by Advanced AI | Enterprise Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    async def render_analytics_sidebar(self):
        """Render analytics sidebar"""
        with st.sidebar:
            st.markdown("### 📊 Analytics Dashboard")
            
            try:
                metrics = await self.analytics.get_dashboard_metrics()
                
                if metrics.get('documents'):
                    st.markdown("#### Document Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Documents", metrics['documents'].get('total_documents', 0))
                        st.metric("Processed", metrics['documents'].get('processed', 0))
                    with col2:
                        st.metric("Pending", metrics['documents'].get('pending', 0))
                        st.metric("Avg Pages", f"{metrics['documents'].get('avg_pages', 0):.1f}")
                
                if metrics.get('conversations'):
                    st.markdown("#### Conversation Metrics")
                    conv = metrics['conversations']
                    st.metric("Total Queries", conv.get('total_conversations', 0))
                    st.metric("Avg Confidence", f"{conv.get('avg_confidence', 0):.1%}")
                    st.metric("Avg Response Time", f"{conv.get('avg_response_time', 0):.2f}s")
                
                if metrics.get('system_health'):
                    st.markdown("#### System Health")
                    health = metrics['system_health']
                    
                    # System status indicator
                    status = health.get('status', 'unknown')
                    status_color = {
                        'healthy': '🟢',
                        'warning': '🟡',
                        'critical': '🔴',
                        'unknown': '⚪'
                    }
                    st.markdown(f"**Status**: {status_color.get(status, '⚪')} {status.title()}")
                    
                    if 'cpu_usage_percent' in health:
                        st.progress(health['cpu_usage_percent'] / 100)
                        st.caption(f"CPU: {health['cpu_usage_percent']:.1f}%")
                    
                    if 'memory_usage_percent' in health:
                        st.progress(health['memory_usage_percent'] / 100)
                        st.caption(f"Memory: {health['memory_usage_percent']:.1f}%")
                
            except Exception as e:
                st.error("Failed to load analytics")
                self.logger.error("Sidebar analytics failed", exception=e)
    
    def render_file_uploader_advanced(self):
        """Advanced file uploader with preview and validation"""
        st.markdown("### 📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF document",
            type=["pdf"],
            accept_multiple_files=False,
            help=f"Maximum file size: {self.config.MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
        
        if uploaded_file:
            # File information
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            with col2:
                st.metric("File Type", uploaded_file.type)
            with col3:
                st.metric("File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
            
            # Validation
            if uploaded_file.size > self.config.MAX_FILE_SIZE:
                st.error(f"File too large! Maximum size is {self.config.MAX_FILE_SIZE / (1024*1024):.0f}MB")
                return None
            
            return uploaded_file
        
        return None
    
    def render_processing_progress(self, stages: List[str], current_stage: int):
        """Render processing progress with stages"""
        st.markdown("### 🔄 Processing Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, stage in enumerate(stages):
            if i <= current_stage:
                progress = (i + 1) / len(stages)
                progress_bar.progress(progress)
                status_text.text(f"Stage {i+1}/{len(stages)}: {stage}")
                
                if i < current_stage:
                    st.success(f"✅ {stage}")
                elif i == current_stage:
                    with st.spinner(f"Processing: {stage}"):
                        time.sleep(0.5)  # Visual feedback
    
    def render_chat_interface_advanced(self):
        """Advanced chat interface with features"""
        st.markdown("### 💬 Intelligent Q&A")
        
        # Search type selection
        search_type = st.selectbox(
            "Search Mode",
            ["semantic", "keyword", "hybrid"],
            help="Choose how to search through your documents"
        )
        
        # Chat input with voice support (if enabled)
        if self.config.ENABLE_VOICE_INTERFACE:
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.chat_input("Ask a question about your document...")
            with col2:
                if st.button("🎤", help="Voice input (coming soon)"):
                    st.info("Voice input feature coming soon!")
        else:
            user_input = st.chat_input("Ask a question about your document...")
        
        return user_input, search_type
    
    def render_response_with_citations(self, response_data: Dict[str, Any]):
        """Render response with advanced formatting and citations"""
        if not response_data:
            return
        
        # Main response
        st.markdown(response_data['response'])
        
        # Response metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = response_data.get('confidence_score', 0)
            confidence_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
            st.metric("Confidence", f"{confidence:.1%}", help="AI confidence in the response")
        
        with col2:
            response_time = response_data.get('response_time', 0)
            st.metric("Response Time", f"{response_time:.2f}s")
        
        with col3:
            context_count = response_data.get('context_used', 0)
            st.metric("Sources Used", context_count)
        
        # Citations
        if response_data.get('citations'):
            with st.expander("📚 Sources & Citations", expanded=False):
                for citation in response_data['citations']:
                    st.markdown(f"""
                    **Source {citation['id']}** (Relevance: {citation['score']:.1%})
                    > {citation['preview']}
                    """)
        
        # Follow-up questions
        if response_data.get('follow_up_questions'):
            st.markdown("### 🤔 Related Questions")
            for i, question in enumerate(response_data['follow_up_questions'], 1):
                if st.button(question, key=f"followup_{i}"):
                    st.session_state['pending_question'] = question
                    st.rerun()
    
    def render_document_insights(self, insights: Dict[str, Any]):
        """Render comprehensive document insights"""
        if not insights:
            return
        
        st.markdown("### 🧠 Document Intelligence")
        
        tabs = st.tabs(["📋 Summary", "📊 Analytics", "🏷️ Entities", "🎯 Topics", "📈 Complexity"])
        
        with tabs[0]:  # Summary
            summary = insights.get('summary', {})
            if summary:
                st.markdown("#### Executive Summary")
                st.info(summary.get('ai_summary', 'No summary available'))
                
                if summary.get('extractive_summary'):
                    with st.expander("📄 Key Excerpts"):
                        st.markdown(summary['extractive_summary'])
        
        with tabs[1]:  # Analytics
            metrics = insights.get('key_metrics', {})
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Words", f"{metrics.get('word_count', 0):,}")
                with col2:
                    st.metric("Reading Time", f"{metrics.get('estimated_reading_time_minutes', 0):.0f} min")
                with col3:
                    st.metric("Unique Keywords", metrics.get('unique_keywords', 0))
                with col4:
                    st.metric("Topics", metrics.get('identified_topics', 0))
        
        with tabs[2]:  # Entities
            entities = insights.get('advanced_analysis', {}).get('entities', [])
            if entities:
                entity_df = pd.DataFrame(entities)
                if not entity_df.empty:
                    st.dataframe(entity_df, use_container_width=True)
            else:
                st.info("No named entities detected")
        
        with tabs[3]:  # Topics
            topics = insights.get('advanced_analysis', {}).get('topics', [])
            if topics:
                for topic in topics:
                    st.markdown(f"**Topic {topic['topic_id'] + 1}** ({topic['document_count']} sections)")
                    st.markdown(f"*Keywords: {', '.join(topic['top_terms'][:5])}*")
                    st.progress(topic['strength'])
                    st.markdown("---")
            else:
                st.info("No distinct topics identified")
        
        with tabs[4]:  # Complexity
            complexity = insights.get('complexity_analysis', {})
            if complexity:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Complexity Score", f"{complexity.get('complexity_score', 0):.1f}/10")
                    st.metric("Avg Sentence Length", f"{complexity.get('average_sentence_length', 0):.1f} words")
                with col2:
                    if 'flesch_reading_ease' in complexity:
                        reading_ease = complexity['flesch_reading_ease']
                        if reading_ease >= 90:
                            level = "Very Easy"
                        elif reading_ease >= 80:
                            level = "Easy"
                        elif reading_ease >= 70:
                            level = "Fairly Easy"
                        elif reading_ease >= 60:
                            level = "Standard"
                        elif reading_ease >= 50:
                            level = "Fairly Difficult"
                        elif reading_ease >= 30:
                            level = "Difficult"
                        else:
                            level = "Very Difficult"
                        
                        st.metric("Reading Level", level)
                        st.metric("Flesch Score", f"{reading_ease:.0f}")
    
    def render_error_handler(self, error: Exception, context: str = ""):
        """Advanced error handling with user-friendly messages"""
        error_type = type(error).__name__
        
        user_friendly_errors = {
            "FileNotFoundError": "The requested file could not be found. Please try uploading again.",
            "MemoryError": "The document is too large to process. Please try a smaller file.",
            "ConnectionError": "Unable to connect to AI services. Please check your internet connection.",
            "TimeoutError": "The operation took too long. Please try again with a smaller document.",
            "ValueError": "There was an issue with the document format. Please ensure it's a valid PDF.",
            "PermissionError": "Permission denied. Please check file permissions and try again."
        }
        
        user_message = user_friendly_errors.get(error_type, f"An unexpected error occurred: {str(error)}")
        
        st.error(f"❌ {user_message}")
        
        with st.expander("🔍 Technical Details (for support)"):
            st.code(f"""
Error Type: {error_type}
Context: {context}
Message: {str(error)}
Timestamp: {datetime.utcnow().isoformat()}
            """)
            
            if st.button("📋 Copy Error Details"):
                # In a real app, you'd implement clipboard functionality
                st.info("Error details copied to clipboard (functionality to be implemented)")


# ================== MAIN APPLICATION CLASS ==================

class ParallaxApp:
    """Main application orchestrator with enterprise features"""
    
    def __init__(self):
        self.config = AdvancedConfig()
        self.logger = EnterpriseLogger(__name__, self.config)
        self.db = DatabaseManager()
        self.processor = AdvancedDocumentProcessor(self.config)
        self.ui = ModernUI(self.config)
        self.analytics = AnalyticsEngine(self.config)
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize comprehensive session state"""
        defaults = {
            'initialized': True,
            'session_id': str(uuid.uuid4()),
            'documents_processed': {},
            'chat_history': [],
            'current_document_id': None,
            'processing_status': 'idle',
            'user_preferences': {
                'theme': 'dark',
                'search_type': 'semantic',
                'show_confidence': True,
                'show_citations': True
            },
            'analytics_enabled': self.config.ENABLE_ANALYTICS,
            'error_history': [],
            'performance_metrics': {
                'total_queries': 0,
                'avg_response_time': 0,
                'successful_uploads': 0
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    async def run(self):
        """Main application entry point"""
        try:
            # Setup UI
            self.ui.setup_page_config()
            self.ui.apply_modern_styling()
            
            # Log session start
            self.analytics.log_user_action('session_started', st.session_state.session_id)
            
            # Render main interface
            await self._render_main_interface()
            
        except Exception as e:
            self.logger.error("Application runtime error", exception=e)
            self.ui.render_error_handler(e, "Main application loop")
    
    async def _render_main_interface(self):
        """Render the main application interface"""
        # Header
        self.ui.render_header()
        
        # Main content area
        main_col, sidebar_col = st.columns([3, 1])
        
        with main_col:
            await self._render_main_content()
        
        with sidebar_col:
            await self.ui.render_analytics_sidebar()
    
    async def _render_main_content(self):
        """Render main content area"""
        # Document upload section
        uploaded_file = self.ui.render_file_uploader_advanced()
        
        if uploaded_file and uploaded_file.name not in st.session_state.documents_processed:
            await self._handle_document_upload(uploaded_file)
        
        # Document processing status
        if st.session_state.documents_processed:
            current_doc = st.session_state.get('current_document_id')
            if current_doc:
                doc_data = st.session_state.documents_processed[current_doc]
                
                # Document insights
                if doc_data.get('insights'):
                    self.ui.render_document_insights(doc_data['insights'])
                
                # Chat interface
                await self._render_chat_section(current_doc)
        else:
            self._render_welcome_section()
    
    def _render_welcome_section(self):
        """Render welcome section for new users"""
        st.markdown("""
        ## Welcome to Project Parallax! 🚀
        
        Transform your document analysis experience with our advanced AI-powered platform.
        
        ### ✨ Key Features:
        
        - 🧠 **Intelligent Document Processing**: Advanced AI understands context, structure, and meaning
        - 🔍 **Multi-Modal Search**: Semantic, keyword, and hybrid search capabilities  
        - 📊 **Deep Analytics**: Comprehensive insights, topic modeling, and entity extraction
        - 💬 **Smart Q&A**: Natural language queries with confidence scoring and citations
        - 📈 **Real-Time Monitoring**: Performance metrics and system health tracking
        - 🔒 **Enterprise Security**: Audit logging, encryption, and compliance features
        
        ### 🚀 Get Started:
        1. Upload a PDF document using the file uploader above
        2. Wait for our AI to process and analyze your document
        3. Start asking questions and explore insights!
        
        ### 💡 Pro Tips:
        - Try different search modes (semantic vs keyword) for different types of questions
        - Use the document insights to understand structure and complexity
        - Check confidence scores to gauge answer reliability
        - Explore follow-up questions for deeper analysis
        """)
    
    async def _handle_document_upload(self, uploaded_file):
        """Handle comprehensive document upload and processing"""
        try:
            document_id = str(uuid.uuid4())
            file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
            
            # Check if document already exists
            with self.db.get_connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM documents WHERE content_hash = ?", 
                    (file_hash,)
                ).fetchone()
                
                if existing:
                    st.warning("📄 This document has already been processed!")
                    st.session_state.current_document_id = existing['id']
                    return
            
            # Save file and create database entry
            file_path = await self._save_uploaded_file(uploaded_file, document_id)
            
            await self._create_document_record(
                document_id, uploaded_file.name, file_path, file_hash, uploaded_file.size
            )
            
            # Processing stages
            stages = [
                "Saving document",
                "Extracting content", 
                "Advanced analysis",
                "Multi-modal processing",
                "Vector indexing",
                "Generating insights"
            ]
            
            progress_container = st.container()
            
            with progress_container:
                self.ui.render_processing_progress(stages, -1)
                
                # Process document
                processing_result = await self.processor.process_document_comprehensive(
                    file_path, document_id
                )
                
                # Update session state
                st.session_state.documents_processed[uploaded_file.name] = processing_result
                st.session_state.current_document_id = document_id
                st.session_state.performance_metrics['successful_uploads'] += 1
                
                # Log success
                self.analytics.log_user_action(
                    'document_processed',
                    st.session_state.session_id,
                    document_id=document_id,
                    processing_time=processing_result.get('processing_time', 0)
                )
            
            st.success("🎉 Document processed successfully! You can now start asking questions.")
            st.rerun()
            
        except Exception as e:
            self.logger.error("Document upload failed", exception=e)
            self.ui.render_error_handler(e, "Document upload and processing")
            
            # Log error
            self.analytics.log_user_action(
                'document_processing_failed',
                st.session_state.session_id,
                error=str(e),
                document_name=uploaded_file.name
            )
    
    async def _save_uploaded_file(self, uploaded_file, document_id: str) -> Path:
        """Save uploaded file to storage"""
        file_path = self.config.PDF_STORAGE_PATH / f"{document_id}_{uploaded_file.name}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(uploaded_file.getbuffer())
        
        return file_path
    
    async def _create_document_record(self, document_id: str, title: str, 
                                    file_path: Path, content_hash: str, file_size: int) -> None:
        """Create document record in database"""
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO documents 
                (id, title, file_path, content_hash, file_size, processing_status)
                VALUES (?, ?, ?, ?, ?, 'processing')
            """, (document_id, title, str(file_path), content_hash, file_size))
            conn.commit()
    
    async def _render_chat_section(self, document_id: str):
        """Render advanced chat section"""
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and isinstance(message["content"], dict):
                    self.ui.render_response_with_citations(message["content"])
                else:
                    st.markdown(message["content"])
        
        # Handle pending question (from follow-up buttons)
        if 'pending_question' in st.session_state:
            user_input = st.session_state.pending_question
            del st.session_state.pending_question
        else:
            # Chat input
            user_input, search_type = self.ui.render_chat_interface_advanced()
        
        # Process user input
        if user_input:
            await self._process_user_query(user_input, document_id, search_type)
    
    async def _process_user_query(self, query: str, document_id: str, search_type: str = "semantic"):
        """Process user query with advanced features"""
        try:
            start_time = time.time()
            
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Search for relevant documents
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching and analyzing..."):
                    search_results = await self.processor.intelligent_search(
                        query, document_id, search_type
                    )
                    
                    if not search_results:
                        st.warning("No relevant information found in the document for your query.")
                        return
                    
                    # Generate enhanced response
                    response_data = await self.processor.generate_enhanced_response(
                        query, search_results
                    )
                    
                    # Render response
                    self.ui.render_response_with_citations(response_data)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_data
                    })
                    
                    # Update performance metrics
                    query_time = time.time() - start_time
                    st.session_state.performance_metrics['total_queries'] += 1
                    
                    current_avg = st.session_state.performance_metrics['avg_response_time']
                    total_queries = st.session_state.performance_metrics['total_queries']
                    new_avg = ((current_avg * (total_queries - 1)) + query_time) / total_queries
                    st.session_state.performance_metrics['avg_response_time'] = new_avg
                    
                    # Log query analytics
                    self.analytics.log_user_action(
                        'query_processed',
                        st.session_state.session_id,
                        query=query,
                        search_type=search_type,
                        response_time=query_time,
                        confidence=response_data.get('confidence_score', 0),
                        document_id=document_id
                    )
                    
        except Exception as e:
            self.logger.error("Query processing failed", query=query, exception=e)
            st.error(f"❌ Sorry, I encountered an error while processing your question: {str(e)}")
            
            # Log error
            self.analytics.log_user_action(
                'query_failed',
                st.session_state.session_id,
                query=query,
                error=str(e)
            )


# ================== APPLICATION ENTRY POINT ==================

async def main():
    """Main application entry point with error handling"""
    try:
        app = ParallaxApp()
        await app.run()
    except Exception as e:
        st.error(f"🚨 Critical application error: {str(e)}")
        st.code(f"Error details: {type(e).__name__}: {str(e)}")
        
        # Emergency fallback UI
        st.markdown("""
        ### 🔧 Troubleshooting
        1. Refresh the page
        2. Clear browser cache
        3. Check internet connection
        4. Ensure all dependencies are installed
        
        If the problem persists, please contact support with the error details above.
        """)


# ================== STARTUP EXECUTION ==================

if __name__ == "__main__":
    # Initialize configuration and check dependencies
    try:
        config = AdvancedConfig()
        
        # Display startup info
        st.sidebar.markdown(f"""
        ### ⚙️ System Info
        - **Version**: {config.VERSION}
        - **Advanced Features**: {'✅' if ADVANCED_FEATURES else '❌'}
        - **Multi-modal**: {'✅' if config.ENABLE_MULTIMODAL else '❌'}
        - **Analytics**: {'✅' if config.ENABLE_ANALYTICS else '❌'}
        """)
        
        # Run the application
        asyncio.run(main())
        
    except Exception as e:
        st.error(f"Failed to start application: {str(e)}")
        st.stop()

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import tempfile
import hashlib
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid
from datetime import datetime
import json

# Document processing libraries
import PyPDF2
from docx import Document
from pptx import Presentation
import pytesseract
from PIL import Image
import io
import email
from email.mime.text import MIMEText
import zipfile

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a processed document chunk"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class DocumentProcessor:
    """Handles multi-format document processing and ingestion"""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = 1000  # ~1KB chunks
        self.overlap = 200

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""

    def extract_text_from_pptx(self, file_path: str) -> str:
        """Extract text from PPTX files"""
        try:
            prs = Presentation(file_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "has_text_frame") and shape.has_text_frame and shape.text_frame.text:
                        text += shape.text_frame.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PPTX text: {e}")
            return ""

    def extract_text_from_csv(self, file_path: str) -> str:
        """Extract text from CSV files"""
        try:
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
            return text
        except Exception as e:
            logger.error(f"Error extracting CSV text: {e}")
            return ""

    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting image text: {e}")
            return ""

    def extract_text_from_email(self, file_path: str) -> str:
        """Extract text from email files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file)
                text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if isinstance(payload, bytes):
                                text += payload.decode('utf-8')
                            elif isinstance(payload, str):
                                text += payload
                else:
                    payload = msg.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        text = payload.decode('utf-8')
                    elif isinstance(payload, str):
                        text = payload
                return text
        except Exception as e:
            logger.error(f"Error extracting email text: {e}")
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Route to appropriate extraction method based on file extension"""
        file_extension = Path(file_path).suffix.lower()

        extraction_methods = {
            '.pdf': self.extract_text_from_pdf,
            '.docx': self.extract_text_from_docx,
            '.pptx': self.extract_text_from_pptx,
            '.csv': self.extract_text_from_csv,
            '.jpg': self.extract_text_from_image,
            '.jpeg': self.extract_text_from_image,
            '.png': self.extract_text_from_image,
            '.eml': self.extract_text_from_email,
            '.txt': lambda path: open(path, 'r', encoding='utf-8').read()
        }

        if file_extension in extraction_methods:
            return extraction_methods[file_extension](file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return ""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks with metadata"""
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            # Create unique chunk ID
            chunk_id = hashlib.md5(f"{metadata['filename']}_{i}_{chunk_text[:50]}".encode()).hexdigest()

            chunk_metadata = {
                **metadata,
                'chunk_index': i // (self.chunk_size - self.overlap),
                'chunk_start': i,
                'chunk_end': min(i + self.chunk_size, len(words)),
                'timestamp': datetime.now().isoformat()
            }

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            ))

        return chunks

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        return chunks

    def process_document(self, file_path: str, document_type: str = "insurance_policy") -> List[DocumentChunk]:
        """Process a single document and return chunks with embeddings"""
        try:
            # Extract text
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []

            # Create metadata
            metadata = {
                'filename': Path(file_path).name,
                'filepath': file_path,
                'document_type': document_type,
                'file_size': os.path.getsize(file_path),
                'processed_at': datetime.now().isoformat(),
                'text_length': len(text)
            }

            # Chunk text
            chunks = self.chunk_text(text, metadata)

            # Generate embeddings
            chunks = self.generate_embeddings(chunks)

            logger.info(f"Processed {len(chunks)} chunks from {Path(file_path).name}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return []

class AstraDBIngester:
    """Handles ingestion of processed chunks into Astra DB"""

    def __init__(self, token: str, database_id: str, keyspace: str = "insurance_advisor"):
        self.token = token
        self.database_id = database_id
        self.keyspace = keyspace
        self.session = None
        self.connect()

    def connect(self):
        """Establish connection to Astra DB"""
        try:
            cloud_config = {
                'secure_connect_bundle': f'secure-connect-{self.database_id}.zip'
            }

            auth_provider = PlainTextAuthProvider('token', self.token)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            self.session = cluster.connect()

            # Create keyspace if it doesn't exist
            if self.session:
                self.session.execute(f"""
                    CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                    WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
                """)

                self.session.set_keyspace(self.keyspace)

                # Create tables
                self.create_tables()

                logger.info("Connected to Astra DB successfully")
            else:
                logger.error("Failed to connect to Astra DB: session is None")
        except Exception as e:
            logger.error(f"Failed to connect to Astra DB: {e}")
            raise

    def create_tables(self):
        """Create necessary tables for document storage"""
        try:
            if self.session:
                # Document chunks table
                self.session.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id text PRIMARY KEY,
                        content text,
                        embedding list<float>,
                        metadata text,
                        document_type text,
                        filename text,
                        created_at timestamp
                    )
                """)

                # Conversation history table
                self.session.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        conversation_id text,
                        message_id text,
                        timestamp timestamp,
                        user_message text,
                        bot_response text,
                        context text,
                        PRIMARY KEY (conversation_id, message_id)
                    )
                """)

                logger.info("Tables created successfully")
            else:
                logger.error("Cannot create tables: session is None")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def ingest_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Ingest document chunks into Astra DB"""
        try:
            if self.session:
                insert_stmt = self.session.prepare("""
                    INSERT INTO document_chunks (chunk_id, content, embedding, metadata, document_type, filename, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """)

                for chunk in chunks:
                    self.session.execute(insert_stmt, [
                        chunk.chunk_id,
                        chunk.content,
                        chunk.embedding,
                        json.dumps(chunk.metadata),
                        chunk.metadata.get('document_type', 'unknown'),
                        chunk.metadata.get('filename', 'unknown'),
                        datetime.now()
                    ])

                logger.info(f"Ingested {len(chunks)} chunks into Astra DB")
                return True
            else:
                logger.error("Cannot ingest chunks: session is None")
                return False
        except Exception as e:
            logger.error(f"Error ingesting chunks: {e}")
            return False

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database"""
        try:
            if self.session:
                result = self.session.execute("SELECT COUNT(*) FROM document_chunks")
                return result.one()[0]
            else:
                logger.error("Cannot get chunk count: session is None")
                return 0
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0

def ingest_documents(file_paths: List[str], astra_config: Dict[str, str]) -> bool:
    """Main function to ingest multiple documents"""
    try:
        # Initialize processor and ingester
        processor = DocumentProcessor()
        ingester = AstraDBIngester(
            token=astra_config['token'],
            database_id=astra_config['database_id'],
            keyspace=astra_config.get('keyspace', 'insurance_advisor')
        )

        all_chunks = []

        # Process each document
        for file_path in file_paths:
            logger.info(f"Processing document: {file_path}")
            chunks = processor.process_document(file_path)
            all_chunks.extend(chunks)

        # Ingest all chunks
        if all_chunks:
            success = ingester.ingest_chunks(all_chunks)
            if success:
                logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(file_paths)} documents")
                return True

        return False

    except Exception as e:
        logger.error(f"Error in document ingestion: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    astra_config = {
        'token': os.getenv('ASTRA_DB_APPLICATION_TOKEN'),
        'database_id': os.getenv('ASTRA_DB_ID'),
        'keyspace': 'insurance_advisor'
    }

    # Example file paths
    file_paths = [
        'data/documents/policy1.pdf',
        'data/documents/policy2.docx',
        'data/documents/terms.pptx'
    ]

    success = ingest_documents(file_paths, astra_config)
    print(f"Ingestion {'successful' if success else 'failed'}")
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logging_config import logger
from config.settings import settings

@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_index: int
    char_count: int

class DocumentChunker:
    def __init__(self, chunk_size=None, chunk_overlap=None, separators=None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        logger.info(f"Initialized chunker with size {self.chunk_size} and overlap {self.chunk_overlap}")

    def chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        content = document.get("content", "")
        if not content.strip():
            logger.warning(f"Empty document: {document.get('filename')}")
            return []
        cleaned = self._preprocess_content(content)
        chunks = self.text_splitter.split_text(cleaned)
        res = []
        for i, cnt in enumerate(chunks):
            res.append(DocumentChunk(
                chunk_id=f"{document['filename']}|{i}",
                content=cnt,
                metadata={
                    **document.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(cnt),
                    "source_filename": document["filename"],
                    "source_file_type": document.get("file_type", "unknown")
                },
                source=document["filename"],
                chunk_index=i,
                char_count=len(cnt)
            ))
        logger.info(f"Chunked {document['filename']} into {len(res)} chunks")
        return res

    def _preprocess_content(self, content: str) -> str:
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\|]', '', content)
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        content = re.sub(r'\n\s*\n', '\n\n', content)
        return content.strip()

document_chunker = DocumentChunker()

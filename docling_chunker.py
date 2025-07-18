from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

converter = DocumentConverter()
doc = converter.convert("document_store/pdfs/P3L_srs_mcp-draft1C.pdf").document

tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"))
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

chunks = list(chunker.chunk(doc))
for c in chunks:
    print(c.meta.headings, "::", c.text)

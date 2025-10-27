"""
Central configuration file for all folder and file paths used in the RAG system.
This ensures consistency across all modules and makes it easy to update paths.
"""

import os

MODEL_NAME="gpt-5-nano"
MODEL_PROVIDER="openai"

RAW_DATA_DIR = os.path.join("data", "raw")
DOCX_DIR = os.path.join(RAW_DATA_DIR, "docx")
PDF_DIR = os.path.join(RAW_DATA_DIR, "pdf")

SAMPLES_DATA_DIR = "data/samples"

PROCESSED_DATA_DIR = "data/processed"

CHROMA_PATH = "chroma"
BM25_DIR = "bm25"
BM25_PATH = os.path.join(BM25_DIR, "bm25_index.pkl")

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = "text-embedding-3-large"

RAG_PROMPT = """Use the following context to answer the question. If you cannot find the answer in the context, say "I don't know."

<contexts>
{context}
</contexts>

<question>
{query}
</question>"""

CHUNKING_PROMPT = """
You are an assistant specialized in splitting text into semantically consistent sections.
 
 <instructions>
    <instruction>The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number</instruction>
    <instruction>Identify points where splits should occur, such that consecutive chunks of similar themes stay together</instruction>
    <instruction>If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2</instruction>
    <instruction>The chunks must be listed in ascending order</instruction>
    <instruction>Each chunk must be between 1000 and 1200 words</instruction>
    <instruction>Make sure that forms, images, tables are in separate chunks along with the text that describes them</instruction>
</instructions>
 
This is the document text:
<document>
{document_text}
</document>
 
Respond with a list of the IDs of the chunks where you believe a split should occur.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
""".strip()

CONTEXTUALIZER_PROMPT = """
You are an assistant specialized in analyzing document chunks and providing relevant context.
 
<instructions>
    <instruction>You will be given a document and a specific chunk from that document</instruction>
    <instruction>Provide 2-3 concise sentences that situate this chunk within the broader document</instruction>
    <instruction>Identify the main topic or concept discussed in the chunk</instruction>
    <instruction>Include relevant information or comparisons from the broader document context</instruction>
    <instruction>Note how this information relates to the overall theme or purpose of the document if applicable</instruction>
    <instruction>Include key figures, dates, or percentages that provide important context</instruction>
    <instruction>Avoid phrases like "This chunk discusses" - instead, directly state the context</instruction>
    <instruction>Keep your response brief and focused on improving search retrieval</instruction>
</instructions>
 
Here is the document:
<document>
{document}
</document>
 
Here is the chunk to contextualize:
<chunk>
{chunk}
</chunk>
 
Respond only with the succinct context for this chunk. Do not mention it is a chunk or that you are providing context.
 
You're provided with the first page of the document. Include a one sentence summary of the document using it at the end of the context.
""".strip()

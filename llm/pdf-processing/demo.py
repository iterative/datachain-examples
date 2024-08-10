from typing import List
from collections.abc import Iterator

from datachain.lib.dc import DataChain, C
from datachain.lib.file import File
from datachain.lib.data_model import DataModel

from unstructured.partition.pdf import partition_pdf

from unstructured.cleaners.core import clean
from unstructured.cleaners.core import replace_unicode_quotes
from unstructured.cleaners.core import group_broken_paragraphs

from unstructured.embed.huggingface import HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingEncoder

# Define the output as a Feature class
class Chunk(DataModel):
    key: str
    text: str
    embeddings: List[float]

# Define embedding encoder

embedding_encoder = HuggingFaceEmbeddingEncoder(
     config=HuggingFaceEmbeddingConfig()
)

# Use signatures to define UDF input/output (these can be pydantic model or regular Python types)
def pdf_chunks(file: File) -> Iterator[Chunk]:
    # Ingest the file
    with file.open() as f:
        chunks = partition_pdf(file=f, chunking_strategy="by_title")

    # Clean the chunks and add new columns
    for chunk in chunks:
        chunk.apply(lambda text: clean(text, bullets=True, extra_whitespace=True, trailing_punctuation=True))
        chunk.apply(replace_unicode_quotes)
        chunk.apply(group_broken_paragraphs)

    # create embeddings
    chunks_embedded = embedding_encoder.embed_documents(chunks)

    # Add new rows to DataChain
    for chunk in chunks_embedded:
        yield Chunk(
            key=file.name.removesuffix("-Paper.pdf"),
            text=chunk.text,
            embeddings_new=chunk.embeddings,
        )

dc = (
    DataChain.from_storage("gs://datachain-demo/neurips")
    .filter(C.name.glob("*.pdf"))
    .gen(document=pdf_chunks)
)

dc

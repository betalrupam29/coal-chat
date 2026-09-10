from pathlib import Path
import pickle

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_PATH = Path("documents/company_policy.pdf")

VECTOR_DB_PATH = Path("vector_db")

INDEX_PATH = VECTOR_DB_PATH / "policy.index"
DATA_PATH = VECTOR_DB_PATH / "policy_data.pkl"


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class PolicyRAG:

    def __init__(self):

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.index = None
        self.documents = []

        VECTOR_DB_PATH.mkdir(
            parents=True,
            exist_ok=True
        )

        if INDEX_PATH.exists() and DATA_PATH.exists():

            self.load_index()

        elif PDF_PATH.exists():

            self.build_index()

        else:

            print(
                f"WARNING: PDF not found at {PDF_PATH}"
            )

    def extract_pdf(self):

        reader = PdfReader(str(PDF_PATH))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

                pages.append({
                    "page": page_number,
                    "text": text
                })

        return pages

    def split_text(
        self,
        pages,
        chunk_size=800,
        overlap=100
    ):

        chunks = []

        for page_data in pages:

            page_number = page_data["page"]
            text = page_data["text"]

            words = text.split()

            start = 0

            while start < len(words):

                end = start + chunk_size

                chunk_words = words[start:end]

                chunk = " ".join(chunk_words)

                if chunk.strip():

                    chunks.append({
                        "text": chunk,
                        "page": page_number,
                        "document": PDF_PATH.name
                    })

                start += chunk_size - overlap

        return chunks

    def build_index(self):

        print("Building RAG index...")

        pages = self.extract_pdf()

        if not pages:

            raise ValueError(
                "No readable text found in the PDF."
            )

        chunks = self.split_text(pages)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype(
            "float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        self.documents = chunks

        faiss.write_index(
            self.index,
            str(INDEX_PATH)
        )

        with open(DATA_PATH, "wb") as f:

            pickle.dump(
                self.documents,
                f
            )

        print(
            f"RAG index created with {len(chunks)} chunks."
        )

    def load_index(self):

        print("Loading existing RAG index...")

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        with open(DATA_PATH, "rb") as f:

            self.documents = pickle.load(f)

        print(
            f"Loaded {len(self.documents)} chunks."
        )

    def search(
        self,
        query: str,
        top_k: int = 4
    ):

        if self.index is None:

            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            document = self.documents[index]

            results.append({
                "text": document["text"],
                "page": document["page"],
                "document": document["document"],
                "score": float(score)
            })

        return results


rag = None


def get_rag():
    global rag

    if rag is None:
        rag = PolicyRAG()

    return rag

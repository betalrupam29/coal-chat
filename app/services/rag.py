from pathlib import Path
import re

from pypdf import PdfReader


PDF_PATH = Path("documents/company_policy.pdf")


class PolicyRAG:

    def __init__(self):

        self.documents = []

        if PDF_PATH.exists():

            self.build_documents()

        else:

            print(
                f"WARNING: PDF not found at {PDF_PATH}"
            )

    def extract_pdf(self):

        reader = PdfReader(
            str(PDF_PATH)
        )

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
        chunk_size=500,
        overlap=50
    ):

        chunks = []

        for page_data in pages:

            page_number = page_data["page"]

            text = page_data["text"]

            words = text.split()

            start = 0

            while start < len(words):

                end = start + chunk_size

                chunk_words = words[
                    start:end
                ]

                chunk = " ".join(
                    chunk_words
                )

                if chunk.strip():

                    chunks.append({

                        "text": chunk,

                        "page": page_number,

                        "document": PDF_PATH.name

                    })

                start += (
                    chunk_size - overlap
                )

        return chunks

    def build_documents(self):

        print(
            "Loading company policy PDF..."
        )

        pages = self.extract_pdf()

        if not pages:

            print(
                "WARNING: No readable text found in PDF."
            )

            return

        self.documents = self.split_text(
            pages
        )

        print(
            f"Loaded {len(self.documents)} policy chunks."
        )

    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"\w+",
            text,
            flags=re.UNICODE
        )

        return set(tokens)

    def search(
        self,
        query: str,
        top_k: int = 4
    ):

        if not self.documents:

            return []

        query_tokens = self.tokenize(
            query
        )

        if not query_tokens:

            return []

        results = []

        for document in self.documents:

            document_tokens = self.tokenize(
                document["text"]
            )

            matched_tokens = (
                query_tokens
                & document_tokens
            )

            if not matched_tokens:

                continue

            # Percentage of query words
            # found in the document
            coverage = (
                len(matched_tokens)
                / len(query_tokens)
            )

            # Extra score for exact phrase
            phrase_bonus = 0

            query_lower = query.lower()

            document_lower = (
                document["text"].lower()
            )

            if query_lower in document_lower:

                phrase_bonus = 0.3

            score = min(
                coverage + phrase_bonus,
                1.0
            )

            results.append({

                "text": document["text"],

                "page": document["page"],

                "document": document["document"],

                "score": score

            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]


rag = None


def get_rag():

    global rag

    if rag is None:

        rag = PolicyRAG()

    return rag

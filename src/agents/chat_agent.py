import re
import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class ChatAgent:
    """
    ChatAgent:
    - Builds a vector store from extracted report text (RAG)
    - Answers user questions in a human-friendly way
    - Avoids repeating raw OCR/pdf text
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", "  ", " ", ""],
        )

        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.model_name = "llama-3.3-70b-versatile"

    # -----------------------------
    # 1) Text Cleaning
    # -----------------------------
    def clean_report_text(self, text: str) -> str:
        """
        Removes repeated headers/footers and non-medical junk commonly found in PDFs:
        - QR code lines, passport, lab IDs, addresses, page numbers, signatures
        """
        if not text:
            return ""

        drop_patterns = [
            r"scan\s*qr",
            r"passport\s*no",
            r"laboratory\s*test\s*report",
            r"this\s+is\s+an\s+electronically\s+authenticated\s+report",
            r"page\s*\d+\s*of\s*\d+",
            r"\bref\.?\s*id\b",
            r"\blab\s*id\b",
            r"\bclient\s*name\b",
            r"\bapproved\s*on\b",
            r"\bprinted\s*on\b",
            r"\bcollected\s*on\b",
            r"\bprocess\s*at\b",
            r"\blocation\b",
            r"\baddress\b",
            r"\bdr\.\b",
            r"\bmd\s*path\b",
            r"\bsignature\b",
        ]

        cleaned_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            low = line.lower()

            # Drop known noisy lines
            if any(re.search(p, low) for p in drop_patterns):
                continue

            # Drop extremely long header-like lines
            if len(line) > 180 and ("scan" in low or "mc-" in low):
                continue

            # Drop repeated separators
            if re.fullmatch(r"[-_]{5,}", line):
                continue

            cleaned_lines.append(line)

        # De-duplicate consecutive duplicate lines
        deduped = []
        prev = None
        for ln in cleaned_lines:
            if ln == prev:
                continue
            deduped.append(ln)
            prev = ln

        return "\n".join(deduped)

    # -----------------------------
    # 2) Vector Store Initialization
    # -----------------------------
    def initialize_vector_store(self, text_content: str):
        """
        Create a FAISS vector store from report text.
        """
        if not text_content or text_content.strip() == "":
            text_content = "No report context available."

        # Clean before splitting/indexing
        cleaned = self.clean_report_text(text_content)
        if not cleaned.strip():
            cleaned = "No report context available."

        texts = self.text_splitter.split_text(cleaned)
        if not texts:
            texts = [cleaned]

        vectorstore = FAISS.from_texts(texts, self.embeddings)
        return vectorstore

    # -----------------------------
    # 3) Chat History Formatting
    # -----------------------------
    def _format_chat_history(self, chat_history):
        """
        chat_history format expected:
        [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        """
        messages = []
        for msg in chat_history:
            if "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        return messages

    # -----------------------------
    # 4) Contextualize Query (ONLY question rewrite)
    # -----------------------------
    def _contextualize_query(self, query: str, chat_history):
        """
        Rewrites the user question into a standalone question.
        DOES NOT analyze report. DOES NOT include report text.
        """
        if not chat_history:
            return query

        recent_history = chat_history[-4:]  # last 2 exchanges
        history_text = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in recent_history
            if "role" in m and "content" in m
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's latest question into a standalone question. "
                    "Do NOT answer. Keep it short and clear."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chat History:\n{history_text}\n\n"
                    f"Latest Question: {query}\n\n"
                    "Standalone Question:"
                ),
            },
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return query

    # -----------------------------
    # 5) Main Response (Human-friendly)
    # -----------------------------
    def get_response(self, query: str, vectorstore, chat_history=None):
        """
        Uses RAG context + Groq LLM to produce a human-friendly medical explanation.
        """
        if chat_history is None:
            chat_history = []

        # 1) Contextualize the question
        contextualized_query = self._contextualize_query(query, chat_history)

        # 2) Retrieve relevant chunks
        context = ""
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(contextualized_query)
            context = "\n\n".join([d.page_content for d in docs if d.page_content])

            # Clean again (safe)
            context = self.clean_report_text(context)

            if context.strip() == "No report context available.":
                context = ""
        except Exception:
            context = ""

        # 3) Prompt for a friendly structured output
        qa_system_prompt = """
You are a medical lab report summarizer.

STRICT RULES:
- NEVER copy/paste raw report text.
- Ignore QR codes, passport numbers, lab IDs, page numbers, doctor names, signatures, addresses.
- Only extract health-relevant test names + values + reference ranges + interpretation.
- If something is missing/unclear, say "Not found in report".
- Be calm, non-alarming, and human-friendly.

OUTPUT FORMAT (ALWAYS):
1) Overall Summary (2-4 lines)
2) Abnormal / Borderline Results (bullets: Test — Value — Range — What it suggests)
3) Normal Highlights (optional, max 5 bullets)
4) Recommended Next Steps (bullets)
5) Lifestyle & Diet Tips (bullets)

If user asks a specific question, answer it within this structure.
"""

        messages = [{"role": "system", "content": qa_system_prompt}]

        # Add recent chat history (optional)
        if chat_history:
            messages.extend(self._format_chat_history(chat_history[-6:]))

        # Provide context + question
        if context.strip():
            user_message = (
                "Here is extracted report context (may be noisy). Use it ONLY to extract facts.\n\n"
                f"Context:\n{context}\n\n"
                f"User Question: {query}"
            )
        else:
            user_message = (
                f"User Question: {query}\n\n"
                "Note: Report context is missing or unreadable. Answer generally and ask for clearer PDF if needed."
            )

        messages.append({"role": "user", "content": user_message})

        # 4) Call Groq
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=700,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

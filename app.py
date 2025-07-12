import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid
from dotenv import load_dotenv
import os
import textwrap

# --- Load environment variables ---
load_dotenv("keys.env")
key = os.getenv("PC_KEY")
print("PC_KEY:", key)  # Debug: should print your API key or None

# --- Pinecone setup ---
pc = Pinecone(api_key=key)
index_name = os.getenv('INDEX')
index = pc.Index(index_name)

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# --- Streamlit UI ---
st.title("📌 Pinecone Upsert Tool (with Sentence Transformers)")

title = st.text_input("Title")
source = st.text_input("Source")
text_input = st.text_area("📝 Text to Embed", height=200)
chunk_min = 200
chunk_max = 500

def chunk_text(text, min_words=200, max_words=500):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        if len(chunk) < min_words and i != 0:
            # append to previous chunk if too small
            chunks[-1].extend(chunk)
            break
        chunks.append(chunk)
        i += max_words
    return [' '.join(chunk) for chunk in chunks]

if st.button("Upsert to Pinecone"):
    if not text_input.strip() or not title or not source:
        st.warning("Please enter text, title, and source.")
    else:
        with st.spinner("Generating embeddings and upserting..."):
            try:
                chunks = chunk_text(text_input, chunk_min, chunk_max)
                entry_id = str(uuid.uuid4())
                for idx, chunk in enumerate(chunks):
                    chunk_id = entry_id if len(chunks) == 1 else f"{entry_id}_{idx+1}"
                    metadata = {
                        "source": source,
                        "title": title,
                        "text": chunk
                    }
                    embedding = model.encode(chunk).tolist()
                    index.upsert([(chunk_id, embedding, metadata)])
                st.success(f"✅ {len(chunks)} chunk(s) embedded and upserted into Pinecone.")
                st.code(f"Entry ID: {entry_id}", language="markdown")
            except Exception as e:
                st.error(f"❌ Error: {e}")

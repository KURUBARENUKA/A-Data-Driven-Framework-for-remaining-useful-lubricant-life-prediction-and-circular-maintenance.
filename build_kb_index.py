import sys
import os

# --- Absolute paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH = os.path.join(PROJECT_ROOT, "agent")

# --- Force agent folder into Python path ---
sys.path.insert(0, AGENT_PATH)

from kb_loader import load_knowledge_base
from embedder import KnowledgeEmbedder

docs = load_knowledge_base(kb_path=os.path.join(PROJECT_ROOT, "knowledge"))
embedder = KnowledgeEmbedder()
embedder.build_index(docs)
embedder.save(os.path.join(AGENT_PATH, "kb.index"))

print("Knowledge base embeddings created successfully.")

"""Regression checks for offline index compatibility and retrieval-only output."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import chromadb
import rag_engine
from embeddings import TfidfEmbedder


class OfflineIndexTests(unittest.TestCase):
    def test_tfidf_index_uses_matching_backend_and_small_pool(self):
        chunks = [
            {'chunk_id':'water', 'source':'Irrigation', 'section':'Water', 'text':'Drip irrigation water emitter'},
            {'chunk_id':'soil', 'source':'Soil', 'section':'Nutrients', 'text':'Nitrogen soil nutrient fertilizer'},
        ]
        embedder = TfidfEmbedder()
        vectors = embedder.encode([c['text'] for c in chunks])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bm25_corpus.json').write_text(json.dumps(chunks))
            (root / 'embedder_backend.json').write_text(json.dumps({'backend':'tfidf-fallback'}))
            client = chromadb.EphemeralClient()
            collection = client.create_collection(rag_engine.COLLECTION_NAME)
            collection.add(ids=[c['chunk_id'] for c in chunks], embeddings=vectors.tolist())
            with patch.object(chromadb, 'PersistentClient', return_value=client), \
                 patch.object(rag_engine, 'DB_DIR', folder), \
                 patch.object(rag_engine, 'BM25_CORPUS_PATH', str(root / 'bm25_corpus.json')), \
                 patch.object(rag_engine, 'DenseEmbedder', side_effect=AssertionError('Unexpected model download')), \
                 patch.dict(os.environ, {'GROQ_API_KEY':'test-placeholder', 'ANTHROPIC_API_KEY':'test-placeholder', 'FARM_RETRIEVAL_ONLY':'1'}):
                engine = rag_engine.RAGEngine()
                self.assertEqual(engine.embedder.backend, 'tfidf-fallback')
                result = engine.answer('drip irrigation')
                self.assertEqual(result['sources'][0]['source'], 'Irrigation')
                self.assertIn('Retrieved source excerpts', result['answer'])
                self.assertNotIn('You are an agricultural advisory assistant', result['answer'])


if __name__ == '__main__':
    unittest.main()

import chromadb
class ResumeVectorStore:
    def __init__(self, collection_name: str = "resumes", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_collection(self, ids: list, embeddings, documents: list):
        """
        (Re)builds the collection from scratch with the given resumes.
        Called fresh each time a new ranking is requested, so old resumes
        from a previous run don't linger.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection didn't exist yet -- fine

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
        )

    def add_chunks(self, ids: list, embeddings, documents: list, metadatas: list):
        """Persist or update resume chunks without replacing other resumes."""
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding, n_results: int) -> list:
        """
        Finds the n_results closest stored embeddings to query_embedding.

        Returns a list of (id, similarity_score) tuples, ChromaDB's own
        ranked order (closest first). similarity_score = 1 - cosine_distance,
        so 1.0 = identical meaning, 0.0 = unrelated.
        """
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]
        return list(zip(ids, similarities))


# Quick manual test: `python vector_store.py`
if __name__ == "__main__":
    from embedder import embed_texts

    docs = {
        "resume_a.txt": "Python developer with 5 years of machine learning experience.",
        "resume_b.txt": "Graphic designer skilled in Photoshop and Illustrator.",
    }
    jd = "Looking for a machine learning engineer skilled in Python."

    embeddings = embed_texts(list(docs.values()))
    jd_embedding = embed_texts([jd])

    store = ResumeVectorStore()
    store.build_collection(ids=list(docs.keys()), embeddings=embeddings, documents=list(docs.values()))
    print(store.query(jd_embedding, n_results=len(docs)))

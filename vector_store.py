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
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass 

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
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding, n_results: int) -> list:
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]
        return list(zip(ids, similarities))


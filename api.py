import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from embedder import embed_texts, semantic_chunk_text
from resume_reader import read_resume
from vector_store import ResumeVectorStore

PROJECT_DIR = Path(__file__).resolve().parent
RESUME_JSON_DIR = PROJECT_DIR / "data" / "resumes"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MODEL_NAMES = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
MAX_FILE_SIZE = 10 * 1024 * 1024

app = FastAPI(title="Resume Screening API")


class MatchedChunk(BaseModel):
	text: str
	similarity: float


class ProcessingError(BaseModel):
	filename: str
	message: str


class ChunkScore(BaseModel):
	chunk_index: int
	similarity: float


class RankedCandidate(BaseModel):
	rank: int
	resume_id: str
	filename: str
	score: float
	json_path: str
	extracted_text: str
	chunks: list[str]
	embeddings: list[list[float]]
	ranked_chunks: list[ChunkScore]
	matched_chunks: list[MatchedChunk]


class RankResponse(BaseModel):
	batch_id: str
	job_description: str
	job_embedding: list[float]
	processed_count: int
	results: list[RankedCandidate]
	errors: list[ProcessingError]


rank_results: dict[str, RankResponse] = {}


@app.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}


@app.get("/models", response_model=list[str])
def get_models() -> list[str]:
	return MODEL_NAMES


@app.post("/rank", response_model=RankResponse)
async def rank_resumes(
	files: list[UploadFile] = File(...),
	job_description: str = Form(...),
	top_n: int = Form(5, ge=1, le=50),
	model_name: str = Form(MODEL_NAMES[0]),
	similarity_threshold: float = Form(0.55, ge=0.30, le=0.90),
) -> RankResponse:
	if not job_description.strip():
		raise HTTPException(status_code=422, detail="Job description cannot be empty.")
	if model_name not in MODEL_NAMES:
		raise HTTPException(status_code=422, detail="Unsupported embedding model.")

	batch_id = uuid4().hex
	job_embedding = embed_texts([job_description], model_name=model_name)[0]
	candidates = []
	processing_errors = []
	all_ids = []
	all_vectors = []
	all_documents = []
	all_metadatas = []

	for uploaded_file in files:
		filename = uploaded_file.filename or "resume"
		try:
			file_bytes = await uploaded_file.read()
			if len(file_bytes) > MAX_FILE_SIZE:
				raise ValueError("File exceeds the 10 MB upload limit.")

			extracted_text = read_resume(filename, file_bytes)
			if not extracted_text:
				raise ValueError("No readable text was found.")

			chunks = semantic_chunk_text(
				extracted_text,
				model_name=model_name,
				similarity_threshold=similarity_threshold,
			)
			if not chunks:
				raise ValueError("No semantic chunks were created.")

			embeddings = embed_texts(chunks, model_name=model_name)
			chunk_scores = embeddings @ job_embedding
			ranked_chunks = sorted(
				enumerate(chunk_scores), key=lambda item: float(item[1]), reverse=True
			)
			top_matches = ranked_chunks[:min(3, len(ranked_chunks))]
			resume_score = float(np.mean([score for _, score in top_matches]))
			resume_id = uuid4().hex

			safe_stem = re.sub(
				r"[^A-Za-z0-9_-]+", "_", Path(filename).stem
			).strip("_") or "resume"
			RESUME_JSON_DIR.mkdir(parents=True, exist_ok=True)
			json_path = RESUME_JSON_DIR / f"{safe_stem}_{resume_id[:8]}.json"
			resume_data = {
				"resume_id": resume_id,
				"source_filename": filename,
				"extracted_at": datetime.now(timezone.utc).isoformat(),
				"extracted_text": extracted_text,
				"model_name": model_name,
				"semantic_chunks": chunks,
			}
			json_path.write_text(
				json.dumps(resume_data, ensure_ascii=False, indent=2),
				encoding="utf-8",
			)

			candidates.append(
				{
					"resume_id": resume_id,
					"filename": filename,
					"score": resume_score,
					"json_path": str(json_path),
					"extracted_text": extracted_text,
					"chunks": chunks,
					"embeddings": embeddings.tolist(),
					"ranked_chunks": [
						{"chunk_index": index, "similarity": float(score)}
						for index, score in ranked_chunks
					],
					"matched_chunks": [
						{
							"text": chunks[index],
							"similarity": float(score),
						}
						for index, score in top_matches
					],
				}
			)
			all_ids.extend(f"{resume_id}-{index}" for index in range(len(chunks)))
			all_vectors.append(embeddings)
			all_documents.extend(chunks)
			all_metadatas.extend(
				{
					"doc_type": "resume_chunk",
					"resume_id": resume_id,
					"source_filename": filename,
					"source_json": str(json_path),
					"chunk_index": index,
					"model_name": model_name,
				}
				for index in range(len(chunks))
			)
		except Exception as error:
			processing_errors.append({"filename": filename, "message": str(error)})

	if candidates:
		all_ids.append(f"job-{batch_id}")
		all_vectors.append(job_embedding.reshape(1, -1))
		all_documents.append(job_description)
		all_metadatas.append(
			{"doc_type": "job_description", "batch_id": batch_id, "model_name": model_name}
		)
		store = ResumeVectorStore(
			collection_name=f"resume_embeddings_{model_name.replace('-', '_')}",
			persist_directory=str(CHROMA_DIR),
		)
		store.add_chunks(
			all_ids,
			np.vstack(all_vectors),
			all_documents,
			all_metadatas,
		)

	candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
	results = [
		{"rank": rank, **candidate}
		for rank, candidate in enumerate(candidates, start=1)
	]
	response = RankResponse(
		batch_id=batch_id,
		job_description=job_description,
		job_embedding=job_embedding.tolist(),
		processed_count=len(candidates),
		results=results,
		errors=processing_errors,
	)
	rank_results[batch_id] = response
	if len(rank_results) > 10:
		rank_results.pop(next(iter(rank_results)))
	return response


@app.get("/rank/{batch_id}", response_model=RankResponse)
def get_rank_result(batch_id: str) -> RankResponse:
	result = rank_results.get(batch_id)
	if result is None:
		raise HTTPException(status_code=404, detail="Ranking result was not found.")
	return result
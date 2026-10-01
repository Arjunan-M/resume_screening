import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from embedder import embed_resume_json, embed_texts
from resume_reader import read_resume
from vector_store import ResumeVectorStore

PROJECT_DIR = Path(__file__).resolve().parent
RESUME_JSON_DIR = PROJECT_DIR / "data" / "resumes"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MODEL_NAMES = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

st.set_page_config(page_title="Resume Screening", layout="wide")
st.title("Resume Screening")


with st.form("resume_upload"):
	uploaded_files = st.file_uploader(
		"Resume files",
		type=["pdf", "docx"],
		accept_multiple_files=True,
	)
	job_description = st.text_area(
		"Job description",
		placeholder="Paste the job description here...",
		height=200,
	)
	top_n = st.number_input(
		"Number of top candidates to show",
		min_value=1,
		value=5,
		step=1,
	)
	model_name = st.selectbox("Embedding model", MODEL_NAMES)
	similarity_threshold = st.slider(
		"Chunk similarity threshold",
		min_value=0.30,
		max_value=0.90,
		value=0.55,
		step=0.05,
		help="A sentence below this cosine similarity starts a new semantic chunk.",
	)
	submitted = st.form_submit_button("Rank candidates")

if submitted:
	st.session_state.pop("last_result", None)
	if not uploaded_files or not job_description.strip():
		st.error("Choose at least one resume and enter a job description first.")
	else:
		candidates = []
		processing_errors = []
		all_ids = []
		all_vectors = []
		all_documents = []
		all_metadatas = []
		batch_id = uuid4().hex

		try:
			with st.spinner("Reading and ranking resumes..."):
				job_embedding = embed_texts([job_description], model_name=model_name)[0]
				for uploaded_file in uploaded_files:
					try:
						extracted_text = read_resume(
							uploaded_file.name, uploaded_file.getvalue()
						)
						if not extracted_text:
							raise ValueError("No readable text was found.")

						resume_id = uuid4().hex
						safe_stem = re.sub(
							r"[^A-Za-z0-9_-]+", "_", Path(uploaded_file.name).stem
						).strip("_") or "resume"
						RESUME_JSON_DIR.mkdir(parents=True, exist_ok=True)
						json_path = RESUME_JSON_DIR / f"{safe_stem}_{resume_id[:8]}.json"
						resume_data = {
							"resume_id": resume_id,
							"source_filename": uploaded_file.name,
							"extracted_at": datetime.now(timezone.utc).isoformat(),
							"extracted_text": extracted_text,
						}
						json_path.write_text(
							json.dumps(resume_data, ensure_ascii=False, indent=2),
							encoding="utf-8",
						)

						resume_data, chunks, embeddings = embed_resume_json(
							json_path,
							model_name=model_name,
							similarity_threshold=similarity_threshold,
						)
						if not chunks:
							raise ValueError("No semantic chunks were created.")
						resume_data["model_name"] = model_name
						resume_data["semantic_chunks"] = chunks
						json_path.write_text(
							json.dumps(resume_data, ensure_ascii=False, indent=2),
							encoding="utf-8",
						)

						chunk_scores = embeddings @ job_embedding
						ranked_chunks = sorted(
							enumerate(chunk_scores),
							key=lambda item: item[1],
							reverse=True,
						)
						top_matches = ranked_chunks[:min(3, len(ranked_chunks))]
						resume_score = float(
							np.mean([score for _, score in top_matches])
						)
						candidates.append(
							{
								"filename": uploaded_file.name,
								"json_path": str(json_path),
								"resume_id": resume_id,
								"extracted_text": extracted_text,
								"chunks": chunks,
								"embeddings": embeddings,
								"ranked_chunks": ranked_chunks,
								"resume_score": resume_score,
							}
						)

						all_ids.extend(
							f"{resume_id}-{index}" for index in range(len(chunks))
						)
						all_vectors.append(embeddings)
						all_documents.extend(chunks)
						all_metadatas.extend(
							{
								"doc_type": "resume_chunk",
								"resume_id": resume_id,
								"source_filename": uploaded_file.name,
								"source_json": str(json_path),
								"chunk_index": index,
								"model_name": model_name,
							}
							for index in range(len(chunks))
						)
					except Exception as error:
						processing_errors.append(f"{uploaded_file.name}: {error}")

			if candidates:
				all_ids.append(f"job-{batch_id}")
				all_vectors.append(job_embedding.reshape(1, -1))
				all_documents.append(job_description)
				all_metadatas.append(
					{
						"doc_type": "job_description",
						"batch_id": batch_id,
						"model_name": model_name,
					}
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
		except Exception as error:
			st.error(f"Could not rank resumes: {error}")
		else:
			if candidates:
				candidates.sort(key=lambda candidate: candidate["resume_score"], reverse=True)
				for rank, candidate in enumerate(candidates, start=1):
					candidate["rank"] = rank
				st.session_state["last_result"] = {
					"candidates": candidates,
					"job_description": job_description,
					"job_embedding": job_embedding,
					"top_n": int(top_n),
				}
				st.success(
					f"Ranked {len(candidates)} resume(s); stored their chunks and the job description in ChromaDB."
				)
		if processing_errors:
			for message in processing_errors:
				st.warning(f"Skipped {message}")
		if not candidates:
			st.error("None of the uploaded resumes could be processed.")

result = st.session_state.get("last_result")
if result:
	candidates = result["candidates"]
	visible_candidates = candidates[:min(result["top_n"], len(candidates))]
	best_candidate = candidates[0]
	st.success(
		f"Top match: {best_candidate['filename']} "
		f"(mean top-3 cosine similarity: {best_candidate['resume_score']:.3f})"
	)
	st.caption(
		"Similarity is a relevance signal, not a hiring decision. Review candidate qualifications directly."
	)
	ranking_rows = [
		{
			"Rank": candidate["rank"],
			"Candidate": candidate["filename"],
			"Mean top-3 cosine similarity": candidate["resume_score"],
		}
		for candidate in visible_candidates
	]
	st.dataframe(pd.DataFrame(ranking_rows), use_container_width=True, hide_index=True)
	if result["top_n"] > len(candidates):
		st.caption(f"Showing all {len(candidates)} successfully processed candidate(s).")

	candidate_options = {
		f"Rank {candidate['rank']}: {candidate['filename']}": candidate
		for candidate in visible_candidates
	}
	selected_label = st.selectbox("Candidate details", list(candidate_options))
	selected_candidate = candidate_options[selected_label]
	st.caption(
		f"JSON: {selected_candidate['json_path']} | "
		f"{len(selected_candidate['chunks'])} semantic chunks"
	)
	with st.expander("Extracted resume text"):
		st.text_area(
			"Extracted text",
			selected_candidate["extracted_text"],
			height=240,
			key=f"resume_text_{selected_candidate['resume_id']}",
		)

	chunks = selected_candidate["chunks"]
	embeddings = selected_candidate["embeddings"]
	st.subheader(f"Chunk matches for {selected_candidate['filename']}")
	st.write(f"{len(chunks)} chunks | {embeddings.shape[1]} dimensions per embedding")
	ranked_rows = [
		{
			"Rank": rank + 1,
			"Cosine similarity": float(score),
			"Resume chunk": chunks[chunk_index],
		}
		for rank, (chunk_index, score) in enumerate(selected_candidate["ranked_chunks"][:10])
	]
	st.dataframe(pd.DataFrame(ranked_rows), use_container_width=True, hide_index=True)

	job_embedding = result["job_embedding"]
	plot_embeddings = embeddings
	plot_labels = [f"Resume chunk {index}" for index in range(1, len(chunks) + 1)]
	plot_types = ["Resume chunk"] * len(chunks)
	plot_texts = chunks.copy()
	plot_embeddings = np.vstack((embeddings, job_embedding))
	plot_labels.append("Job description")
	plot_types.append("Job description")
	plot_texts.append(result["job_description"])
	if len(plot_embeddings) == 1:
		coordinates = [[0.0, 0.0]]
	else:
		coordinates = PCA(n_components=2).fit_transform(plot_embeddings).tolist()

	plot_data = pd.DataFrame(
		{
			"PC1": [point[0] for point in coordinates],
			"PC2": [point[1] for point in coordinates],
			"Label": plot_labels,
			"Type": plot_types,
			"Text": plot_texts,
		}
	)
	figure = px.scatter(
		plot_data,
		x="PC1",
		y="PC2",
		color="Type",
		symbol="Type",
		hover_name="Label",
		hover_data={"Text": True},
		color_discrete_map={"Resume chunk": "#147D92", "Job description": "#E07A35"},
		title="Selected resume and job description embeddings projected to two dimensions",
	)
	st.plotly_chart(figure, use_container_width=True)

	with st.expander("Embedding values"):
		embedding_options = [f"Resume chunk {index}" for index in range(1, len(chunks) + 1)]
		embedding_options.insert(0, "Job description")
		chunk_number = st.selectbox(
			"Embedding to inspect",
			options=embedding_options,
		)
		if chunk_number == "Job description":
			selected_vector = job_embedding
		else:
			selected_vector = embeddings[int(chunk_number.rsplit(" ", 1)[1]) - 1]
		embedding_values = pd.DataFrame(
			{
				"Dimension": list(range(len(selected_vector))),
				"Value": selected_vector,
			}
		)
		st.dataframe(embedding_values, use_container_width=True, hide_index=True)

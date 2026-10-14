import httpx
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

API_BASE_URL = "http://127.0.0.1:8000"
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
		try:
			with st.spinner("Sending resumes to the API and ranking candidates..."):
				files = [
					("files", (item.name, item.getvalue(), item.type))
					for item in uploaded_files
				]
				response = httpx.post(
					f"{API_BASE_URL}/rank",
					files=files,
					data={
						"job_description": job_description,
						"top_n": str(int(top_n)),
						"model_name": model_name,
						"similarity_threshold": str(similarity_threshold),
					},
					timeout=600,
				)
				response.raise_for_status()
				batch_id = response.json()["batch_id"]
				result_response = httpx.get(
					f"{API_BASE_URL}/rank/{batch_id}", timeout=30
				)
				result_response.raise_for_status()
				api_result = result_response.json()

			candidates = [
				{
					"rank": item["rank"],
					"filename": item["filename"],
					"json_path": item["json_path"],
					"resume_id": item["resume_id"],
					"extracted_text": item["extracted_text"],
					"chunks": item["chunks"],
					"embeddings": np.asarray(item["embeddings"]),
					"ranked_chunks": [
						(chunk["chunk_index"], chunk["similarity"])
						for chunk in item["ranked_chunks"]
					],
					"resume_score": item["score"],
				}
				for item in api_result["results"]
			]
			st.session_state["last_result"] = {
				"candidates": candidates,
				"job_description": api_result["job_description"],
				"job_embedding": np.asarray(api_result["job_embedding"]),
				"top_n": int(top_n),
			}
			for error in api_result["errors"]:
				st.warning(f"Skipped {error['filename']}: {error['message']}")
			if candidates:
				st.success(
					f"Ranked {len(candidates)} resume(s) through the API and stored them in ChromaDB."
				)
			else:
				st.error("None of the uploaded resumes could be processed.")
		except httpx.HTTPStatusError as error:
			detail = error.response.text
			try:
				detail = error.response.json().get("detail", detail)
			except ValueError:
				pass
			st.error(f"The API returned an error: {detail}")
		except httpx.RequestError as error:
			st.error(
				f"Could not reach the API at {API_BASE_URL}. Start it with "
				"`uvicorn api:app --reload` and try again."
			)

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

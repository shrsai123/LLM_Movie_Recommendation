import ast
import json
import logging

import pandas as pd
from langchain_classic.schema import Document
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

logger = logging.getLogger(__name__)


def robust_parse(s):
    """Parse stringified JSON/Python literals from TMDB columns."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []


def get_director(crew_str: str) -> str:
    crew = robust_parse(crew_str)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def get_cast(cast_str: str, limit: int = 5) -> str:
    cast_list = robust_parse(cast_str)
    return ", ".join([actor["name"] for actor in cast_list[:limit]])


def load_and_clean(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load raw CSVs, merge, and extract structured fields."""
    logger.info("Loading raw TMDB data")
    df_movies = pd.read_csv(movies_path)
    df_credits = pd.read_csv(credits_path)
    df = df_movies.merge(df_credits, on="title")

    logger.info(f"Merged dataset: {len(df)} movies")

    df["genre_data"] = df["genres"].apply(robust_parse)
    df["keyword_data"] = df["keywords"].apply(robust_parse)
    df["collection_data"] = (
        df["belongs_to_collection"].apply(robust_parse)
        if "belongs_to_collection" in df.columns
        else [{} for _ in range(len(df))]
    )
    df["Genres"] = df["genre_data"].apply(
        lambda values: ", ".join(item["name"] for item in values if item.get("name"))
    )
    df["Keywords"] = df["keyword_data"].apply(
        lambda values: ", ".join(item["name"] for item in values if item.get("name"))
    )
    df["Director"] = df["crew"].apply(get_director)
    df["Cast"] = df["cast"].apply(get_cast)

    df["combined_info"] = df.apply(
        lambda row: (
            f"Type: Movie, Title: {row['title']}, "
            f"Director: {row['Director']}, Cast: {row['Cast']}, "
            f"Released: {row['release_date']}, Genres: {row['Genres']}, "
            f"Keywords: {row['Keywords']}, "
            f"Vote_Average: {row['vote_average']}, "
            f"Description: {row['overview']}"
        ),
        axis=1,
    )

    logger.info(f"Cleaned dataset: {len(df)} movies")
    return df


def build_movie_documents(df: pd.DataFrame) -> list[Document]:
    """Create one searchable FAISS document per movie with ranking metadata."""
    documents = []

    for _, row in df.iterrows():
        collection = row.get("collection_data")
        if not isinstance(collection, dict):
            collection = {}

        release_date = row.get("release_date")
        release_year = (
            str(release_date)[:4]
            if pd.notna(release_date) and str(release_date).strip()
            else None
        )
        overview = row.get("overview")
        overview = str(overview) if pd.notna(overview) else ""

        metadata = {
            "id": int(row["id"]) if pd.notna(row.get("id")) else None,
            "title": str(row["title"]),
            "release_year": release_year,
            "overview": overview,
            "genre_ids": [
                int(item["id"])
                for item in row.get("genre_data", [])
                if item.get("id") is not None
            ],
            "genres": [
                item["name"]
                for item in row.get("genre_data", [])
                if item.get("name")
            ],
            "keyword_ids": [
                int(item["id"])
                for item in row.get("keyword_data", [])
                if item.get("id") is not None
            ],
            "keywords": [
                item["name"]
                for item in row.get("keyword_data", [])
                if item.get("name")
            ],
            "collection_id": collection.get("id"),
            "collection_name": collection.get("name"),
            "vote_average": (
                float(row["vote_average"])
                if pd.notna(row.get("vote_average"))
                else None
            ),
            "ranking_metadata_available": True,
            "candidate_sources": ["faiss"],
        }
        documents.append(Document(page_content=row["combined_info"], metadata=metadata))

    logger.info("Created %s movie-level documents", len(documents))
    return documents


def save_combined_csv(df: pd.DataFrame, output_path: str):
    """Save the combined_info column for downstream loading."""
    df[["combined_info"]].to_csv(output_path, index=False)
    logger.info(f"Saved combined CSV to {output_path}")


def load_and_split_documents(
    csv_path: str, chunk_size: int = 1000, chunk_overlap: int = 30
) -> list[Document]:
    """Load the combined CSV and split into LangChain documents."""
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    data = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    docs = splitter.split_documents(documents=data)
    logger.info(f"Split into {len(docs)} document chunks")
    return docs


def run_preprocessing(config: dict) -> list[Document]:
    """Load the TMDB dataset and create structured movie-level documents."""
    df = load_and_clean(
        movies_path=config["data"]["movies_path"],
        credits_path=config["data"]["credits_path"],
    )
    save_combined_csv(df, config["data"]["combined_csv_path"])
    return build_movie_documents(df)

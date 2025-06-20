import json
import logging
import os

import click
import psycopg
from pgvector.psycopg import register_vector
from psycopg.sql import SQL, Identifier
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default table schema will be created if not exists
TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    doc_year TEXT,
    doc_title TEXT,
    doc_url TEXT,
    chunk_index INTEGER,
    chunk_title TEXT,
    chunk_summary TEXT,
    chunk_content TEXT,
    embedding VECTOR(384)
);
"""


@click.command()
@click.option(
    "--dir",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing JSON chunk files",
)
@click.option(
    "--db-url",
    "-u",
    default=lambda: os.getenv(
        "DATABASE_URL", "postgresql://admin:admin@localhost:5432/wwdc_vector"
    ),
    help="Postgres connection URL or DSN",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="clears the database for fresh embeddings",
)
@click.option(
    "--table", "-t", default="rag_chunks", help="Target Postgres table for embeddings"
)
@click.option(
    "--model",
    "-m",
    default="all-MiniLM-L6-v2",
    help="SentenceTransformer model name for embeddings",
)
def embed_and_load(dir, db_url, reset: bool, table, model):
    """
    Iterate over JSON files under `dir`, compute embeddings for each chunk,
    and insert into Postgres table `table` using pgvector.
    """
    # Load embedding model
    embedder = SentenceTransformer(model, truncate_dim=384)
    logger.info(f"Loaded embedding model: {model}")

    # Connect to Postgres
    with psycopg.connect(db_url) as conn:
        logger.info("Connected to Postgres")
        register_vector(conn)
        with conn.cursor() as cur:
            if reset:
                logger.info(f"Clearning {table}")
                SQL("DROP TABLE IF EXISTS {table}").format(table=Identifier(table))
            # Ensure table exists
            cur.execute(SQL(TABLE_SCHEMA).format(table=Identifier(table)))
            conn.commit()
            logger.info(f"Ensured table '{table}' exists")

            # Process each JSON file
            for filename in os.listdir(dir):
                if not filename.lower().endswith(".json"):
                    continue
                logger.info(f"Processing file: {filename}")
                path = os.path.join(dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                year = doc.get("year")
                title = doc.get("title")
                url = doc.get("url")
                chunks = doc.get("chunks", [])

                for idx, chunk in enumerate(chunks):
                    if chunk.get("type") == "transcript":
                        continue
                    chunk_title = chunk.get("title")
                    summary = chunk.get("summary")
                    content = chunk.get("content", "")

                    # Compute embedding (normalized)
                    embedding = embedder.encode(
                        content,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )

                    # Insert row
                    cur.execute(
                        SQL(
                            "INSERT INTO {table} "
                            "(doc_year, doc_title, doc_url, chunk_index, "
                            "chunk_title, chunk_summary, chunk_content, embedding) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                        ).format(table=Identifier(table)),
                        (
                            year,
                            title,
                            url,
                            idx,
                            chunk_title,
                            summary,
                            content,
                            embedding.tolist(),
                        ),
                    )
        conn.commit()


if __name__ == "__main__":
    embed_and_load()

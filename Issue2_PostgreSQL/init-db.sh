#!/bin/bash
set -e

echo "Running init-db.sh script..."

# Create the pgvector extension and add the vector column if not exists
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
    DO \$\$ BEGIN
        ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS embedding_vector vector(768);
    EXCEPTION
        WHEN undefined_table THEN
            CREATE TABLE image_embeddings (
                id SERIAL PRIMARY KEY,
                image_path TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL,
                embedding FLOAT8[],
                embedding_vector vector(1000)
            );
    END \$\$;
EOSQL

echo "init-db.sh script completed."

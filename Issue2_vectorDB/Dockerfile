FROM postgres:15

# Install necessary packages
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-15 \
    gcc \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone pgvector, build and install
RUN git clone https://github.com/pgvector/pgvector /tmp/pgvector && \
    cd /tmp/pgvector && \
    make && make install && \
    rm -rf /tmp/pgvector

# Copy initialization script
COPY init-db.sh /docker-entrypoint-initdb.d/

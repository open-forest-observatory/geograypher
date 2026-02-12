# Base python image
# TODO fix the cchardet issue and upgrade to a more modern python version
FROM python:3.10-slim
# Install git, gcc/g++, and curl for downloading and compiling
# Also install libgl1 and libglib2.0-0 for cv2/albumentations dependency
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    curl \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the container workdir
WORKDIR /app
# Copy files from current directory into /app
COPY . /app

# Install the module dependencies with poetry without creating a virtual environment
RUN /root/.local/bin/poetry config virtualenvs.create false && /root/.local/bin/poetry install

# Run the detector script
CMD python /app/geograypher/entrypoints/render_labels.py
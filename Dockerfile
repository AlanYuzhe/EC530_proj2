FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml /app/

RUN conda env create -f /app/environment.yml && \
    conda clean -afy

SHELL ["conda", "run", "-n", "ec530-new", "/bin/bash", "-c"]

COPY . /app/

CMD ["python", "app.py"]

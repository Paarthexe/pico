FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COPY sandbox/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt \
    && mkdir /input /output \
    && chmod 755 /input /output
COPY sandbox/worker.py /opt/pico-worker.py
USER 65532:65532
WORKDIR /output
ENTRYPOINT ["python", "-I", "/opt/pico-worker.py"]
CMD ["30"]

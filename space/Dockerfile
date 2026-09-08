# chilecule as a running service.
#
# Docker rather than a plain Python Space because the interesting half of this
# toolkit is not pip-installable: smina (docking) and fpocket (binding-site
# detection) are compiled binaries that come from conda-forge. A Space without
# them can profile molecules but cannot dock anything, which would leave out
# the part that makes the validation harness worth having.

FROM continuumio/miniconda3:24.9.2-0

# Hugging Face Spaces runs the container as uid 1000 and mounts nothing
# writable except /home/user and /tmp. Everything below is arranged to respect
# that rather than to be discovered at runtime.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/opt/conda/bin:/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    CHILECULE_OUTPUT=/tmp/runs

# The compiled tools. Pinned to conda-forge because that is the only channel
# that ships smina and fpocket for linux-64, and an unpinned channel order has
# resolved to a broken mix before.
RUN conda install -y -c conda-forge \
        python=3.11 \
        smina \
        fpocket \
    && conda clean -afy

WORKDIR /app
COPY --chown=user . /app

# The package itself, plus Gradio for the interface. Installed after the conda
# step so a change to the Python code does not invalidate the slow layer.
# gradio 5, not a pinned 4.x.
#
# Pinning gradio 4.44.1 against a moving ecosystem failed twice in a row, for
# unrelated reasons each time: first huggingface_hub 1.0 removed HfFolder,
# which gradio 4 imports at module load; then a newer starlette changed
# TemplateResponse so gradio 4's positional call passed a dict where a template
# name was expected, and every page 500'd. Holding an old gradio in place would
# have meant pinning the hub, starlette and fastapi too, and re-pinning each
# time one of them moved.
#
# gradio 5 is built against the current stack. Every component this app uses --
# Blocks, Tab, Row, Textbox, Checkbox, Button, Markdown, Examples, themes --
# is unchanged between 4 and 5, so the migration touches nothing in app.py.
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "gradio>=5.9,<6"

# /app is created by WORKDIR as root; the process runs as uid 1000 and the
# workflows write reports. Same class of failure as a read-only cache dir, and
# it surfaces as a 500 rather than as anything that names the real cause.
RUN mkdir -p /tmp/runs /tmp/mpl && chown -R user:user /tmp/runs /tmp/mpl /app

USER user
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD ["python", "space/app.py"]

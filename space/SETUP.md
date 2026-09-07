# Deploying this Space

Nothing here is deployed. These are the steps, for when the account exists.

1. Create a free account at https://huggingface.co (no card needed).

2. Create a Space: **New Space** → name `chilecule` → **SDK: Docker** →
   **Public**. Start on the free CPU tier; upgrading to persistent CPU (~$9/mo)
   only buys you "never sleeps".

3. Push this repository to it. The Space needs `Dockerfile` and `README.md` at
   its root, so the `space/` directory is copied up rather than the repo as-is:

   ```bash
   git clone https://huggingface.co/spaces/<user>/chilecule /tmp/hf-chilecule
   cd /tmp/hf-chilecule
   cp -r ~/chilecule/{src,pyproject.toml,LICENSE,NOTICE} .
   cp ~/chilecule/space/{Dockerfile,app.py,README.md} .
   mkdir -p space && cp ~/chilecule/space/app.py space/app.py
   git add -A && git commit -m "chilecule" && git push
   ```

   The first build takes ten minutes or so — conda resolving smina and fpocket
   is the slow part.

## Do not set an Anthropic key on this Space

The agent workflows are not reachable from the interface, so a key would buy
nothing and could only cost money. The tool layer needs no credential of any
kind: ChEMBL and PubChem are keyless public APIs.

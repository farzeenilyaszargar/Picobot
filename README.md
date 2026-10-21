# Picoboi

An experiment in small, personal AI that runs in your browser. Made by **@farilyaz**.

Picoboi explores tiny language models and browser-local tools. The current website runs the locally trained 99,120-parameter FP32 model in a worker. It is a limited 800-word experiment; replies can be wrong. Ternary weights and extra tools remain future experiments.

The next model's [training corpus](training/conversations/README.md) now uses a 968-word vocabulary after category generalization and includes longer conversations. Its [concept parser prototype](training/conversations/PARSER.md) preserves specific words through bound references. This next model is not trained or deployed yet.

## Run the frontend

No dependencies or build step are required. From the repository root:

```sh
python3 -m http.server 4173 --directory web/dist
```

Open [localhost:4173](http://localhost:4173).

## Project files

- [`web/dist/`](web/dist/): HTML, CSS, JavaScript, and favicon.
- [`web/README.md`](web/README.md): frontend details and browser inference.
- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md): model research, browser constraints, training stages, and evaluation plans.
- [`training/README.md`](training/README.md): the Picoboi 100K synthetic chat baseline, frozen 800-word vocabulary, tokenizer, corpus, and audit tools.

The chat streams actual model replies, supports New chat, and explains vocabulary and context limits. Conversation state stays in memory. Weights download from this website; prompts are not sent to an inference service. With **Look up unfamiliar topics** enabled, words outside the model vocabulary trigger a Wikipedia search with a streamed reference excerpt and source links. Extracted search terms are sent to Wikipedia; switch off the option to keep chat input local. Web results do not retrain the model.

## Browser inference

Rebuild browser assets and verify agreement with PyTorch:

```sh
.venv/bin/python training/export_browser.py
node training/tests/browser-inference.mjs
```

The model has a 128-token context including history and generated output. Older conversation pairs are omitted when needed, with a notice. See [web/README.md](web/README.md) and the [model card](training/checkpoints/MODEL_CARD.md).

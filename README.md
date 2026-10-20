# Picoboi

An experiment in small, personal AI that runs in your browser. Made by **@farilyaz**.

Picoboi explores natively trained ternary language models, local tools, memory, and optional specialist models. The current implementation is a minimal white chat frontend with an About page. The language model is not connected yet; the interface labels this clearly.

## Run the frontend

No dependencies or build step are required. From the repository root:

```sh
python3 -m http.server 4173 --directory web/dist
```

Open [localhost:4173](http://localhost:4173).

## Project files

- [`web/dist/`](web/dist/): HTML, CSS, JavaScript, and favicon.
- [`web/README.md`](web/README.md): frontend details and the model adapter interface.
- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md): model research, browser constraints, training stages, and evaluation plans.
- [`training/README.md`](training/README.md): the Picoboi 100K synthetic chat baseline, frozen 800-word vocabulary, tokenizer, corpus, and audit tools.

The frontend includes message submission, conversation starters, a New chat action, and a model adapter hook for future integration. Conversation state stays in memory. There is currently no inference backend or model download.

## Connect a model

Once the inference engine is ready, provide an adapter in the page:

```js
window.picoboi.setModel({
  async generate(messages) {
    return await yourEngine.generate(messages);
  },
});
```

The adapter receives user/assistant messages and returns an answer as a string. Model loading, context limits, cancellation, and streaming will depend on the selected engine.

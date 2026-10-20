# Picoboi frontend

A buildless, responsive chat frontend. The Home and About views share one static page. The frontend uses no remote APIs, external fonts, analytics, or persistent conversation storage.

## Preview locally

From this directory:

```sh
python3 -m http.server 4173 --directory dist
```

Open http://localhost:4173. Use the About link to read the project description and creator credit.

## Connect the model later

The current chat accepts messages and explicitly explains that the model is not connected. It does not simulate an AI response.

The integration hook in `dist/app.js` accepts a model adapter:

```js
window.picoboi.setModel({
  async generate(messages) {
    // Send messages to your browser-local worker/inference engine.
    // Return the generated answer as a string.
    return await yourEngine.generate(messages);
  },
});
```

History is held in memory. New chat clears it and ignores outstanding replies. The model engine remains responsible for cancellation, context limits, loading/unloading, and model-specific prompt templates. The integration can also wrap a future backend if the architecture changes. Streaming can be added when the chosen engine's interface is known.

## Source

- `dist/index.html`: navigation, chat view, About content.
- `dist/styles.css`: white, minimal, responsive styling.
- `dist/app.js`: routing, composer interactions, chat state, model integration hook.

No dependency installation or build step is required.

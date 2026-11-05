# Open sky art direction

The sky backdrop was generated using the built-in ImageGen tool, then saved into the project as `web/dist/assets/open-sky.webp` (68,160 bytes). The original generated PNG remains in Codex's generated-image library. No external image service is needed by the website.

The user's reference guided the blue/cream palette, airy clouds, subtle film grain, and ASCII bird idea. The sky is a new generated image, and the small ASCII birds are original code-native artwork built with `tools/build_ascii_birds.py`. The reference itself is not served by the website.

## Final image-generation prompt

> Create one production website BACKGROUND ASSET only, landscape aspect ratio about 16:10, high resolution. A beautiful open blue daytime sky, softly photographic with a nostalgic analog 35mm-film grain texture. Midtone airy azure/cornflower blue (#438bb9 and #72abd0), a little deeper blue toward the top, with gentle pale ivory wispy cumulus clouds diffusing at the outer left, lower edge and distant right. Clouds should feel soft and natural, out of focus and ethereal, never stormy or gray. Central 60% should be mostly clear blue and visually quiet for a chat interface; avoid strong cloud edges behind center text. Dreamy, generous, simple, calm, soothing. Inspired by a grainy blue sky photograph with a warm cream ASCII bird, but this asset is ONLY THE EMPTY SKY. No birds, no objects, no typography, no text, no UI, no horizon, no sun, no buildings, no mountains. Make the whole image continuous sky to every edge.

## Interface treatment

- Warm cream display type with navy secondary text against the sky.
- Glossy translucent composer: bright rim, layered highlights, blurred backdrop, and a pale blue base.
- Small ASCII gulls with eight wing frames and slow, staggered flights that wrap outside the viewport. Five birds on desktop, four on mobile; lower opacity during conversations.
- Bird animation runs in CSS. A visible pause control, reduced-motion still poses, and off-tab suspension keep the effect unobtrusive.
- Chat bubbles, model picker, suggestions, About page, focus states, and favicon use the same blue/cream palette.

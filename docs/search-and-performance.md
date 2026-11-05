# Search and performance

PicoBoi is a buildless browser app. Search-facing files live in `web/dist` and require no JavaScript to expose titles, descriptions, canonical links, structured data or About content.

## Production origin

Canonical links, social metadata, structured data, `robots.txt`, `sitemap.xml` and `llms.txt` currently use the existing Sites origin:

https://picobot.lumberfizz.chatgpt.site

If a custom domain becomes the primary origin, update all of these together and redirect the old origin. Do not submit localhost to search engines.

## Implemented

- Unique titles, descriptions and self-canonical URLs for `/` and `/about/`.
- Open Graph and Twitter sharing metadata using the existing sky asset.
- Factual WebSite, WebApplication, SoftwareSourceCode and AboutPage entities. No fabricated ratings, reviews, app-store listing or license.
- A visible, static About page explaining model size, behavior, limits, privacy and source code. Legacy `/#about` links redirect to it.
- A sitemap containing actual document URLs and crawl permission in robots.txt.
- A small factual `llms.txt` reference. This is supplemental documentation, not a guarantee of AI search inclusion.
- A web manifest for browser app identity; there is no native app-store listing to optimize.
- Early, high-priority discovery of the 68 KB sky background. System fonts, buildless modules and existing lightweight assets are retained.
- About loads the sky controls only, without downloading or initializing the chat worker and model.

## Indexing and production checks

The connected Sites project required sign-in (unauthenticated HTTP 401) during implementation. Search metadata cannot make an authenticated site publicly indexable. Publishing source does not change its access policy. Public access requires the owner's explicit authorization.

After public access is enabled:

1. Check `/`, `/about/`, `/robots.txt`, `/sitemap.xml` and social images return successful responses without sign-in.
2. Verify the domain in Google Search Console and Bing Webmaster Tools; submit `/sitemap.xml` and inspect both document URLs. These account-verification actions are not performed by this change.
3. Validate structured data and preview social cards against the public URL.
4. Measure the deployed site, where compression, caching and network latency differ from localhost. This change does not assert host cache-header support or field Core Web Vitals results.

## Validation

Local Lighthouse 13.6.0 mobile baseline: Performance 100, Accessibility 100, Best Practices 100, SEO 100. This is a synthetic lab result, not a ranking score. Final measurements are recorded below after validation.

JSON-LD, sitemap XML, manifest JSON, heading count and local file links were checked. Browser verification covered chat submission, About navigation, legacy hash routing and a 390 px mobile layout. Existing sky control tests passed.

## References

- [Google: AI features and your website](https://developers.google.com/search/docs/appearance/ai-features): established SEO practices apply to AI search; no special schema or AI file guarantees visibility.
- [Google: JavaScript SEO basics](https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics): discoverable content and links matter.
- [Bing Webmaster Guidelines](https://www.bing.com/webmasters/help/bing-webmaster-guidelines-30fba23a): use accurate content and structured data, without keyword stuffing or misleading claims.
- [Optimize LCP](https://web.dev/articles/optimize-lcp/): discover important CSS background imagery early.

Final local mobile results: Accessibility 100, Best Practices 100, SEO 100. Performance measured 98 in the first run (LCP 1.0 s, TBT 30 ms, CLS 0.099) and 100 in a trace-enabled follow-up (LCP 0.9 s, TBT 40 ms, CLS 0). The trace contained no layout-shift events; the earlier shift did not reproduce, so no unsupported claim of a fixed root cause is made. Lab timing and layout stability can vary between runs. Reports are retained locally under `previews/seo/` (ignored by Git).

About-page follow-up: the first audit identified insufficient contrast in the lower explanatory text and return link. Both now use darker navy. The corrected About page scored 100 in all four Lighthouse categories, with LCP 0.8 s, TBT 0 ms and CLS 0.

Authorized production checks confirmed HTTP 200 for both documents, robots.txt, sitemap.xml, llms.txt, the manifest, app code and sky asset. Text assets are served with gzip; the already-compressed WebP is served without redundant gzip. Current host responses use `public, max-age=0, must-revalidate`. The crawler files, manifest, app module and sky image matched the source snapshot; Sites transforms the document responses. These service-authorized checks do not establish anonymous crawler access: unauthenticated requests to both pages still return HTTP 401 under access mode `custom`.

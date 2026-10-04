# Verification record

Verified on October 4, 2026, in WSL/Linux with Python 3.12.15 and Node.js 24.21.0.

## Backend

Twelve backend tests passed:

1. CSV ingestion, profiling, paginated preview, authenticated access, and account isolation.
2. Classification model search across linear, tree, ensemble, and neural-network families; three-fold cross-validation; SHAP; error analysis; unseen-category/missing-value prediction; input validation; artifact download; and standalone exported inference.
3. Regression training, SHAP, error diagnostics, and repeatable predictions on new records.
4. Cancelled jobs cannot publish a model.
5. Group-aware and chronological splits maintain separation.
6. Excel upload and scatter visualization data.
7. Production dashboard deep links work without shadowing API routes.
8. Hashed assets use immutable caching; missing assets remain 404.
9. Static file requests cannot read outside the frontend build directory.
10. Automatic clustering detection, target-free training, three-fold validation, SHAP, group profiles, assignment ambiguity, and new-record group prediction.
11. Date feature engineering and standalone exported inference matching API predictions; the export includes the portable feature transformer.
12. IQR outlier profiling and group/time cross-validation separation.

Ruff checks passed for application and test code.

## Browser

All three Chromium end-to-end scenarios passed together against the final single-container production deployment (`3 passed`, approximately two minutes):

- Desktop: demo sign-in → dataset visualization → data preview → evidence-based chat → three-step training wizard → background ML training → cross-validation comparison → SHAP → prediction → ZIP download → theme change. Verification also covered Settings visibility on a short desktop viewport, a direct page reload, same-origin API documentation, and unknown API routes.
- Mobile, 390 × 844: demo sign-in → responsive navigation → dataset exploration → rendered scatter points → no page-wide horizontal overflow.
- Clustering: demo sign-in → choose dataset → select “Find groups” → review and train → group visualization → SHAP → new-record group prediction.

Prediction checks explicitly await the completed API response and verify HTTP 200 before checking the displayed output. This accommodates first-use model loading on the resource-constrained verification machine.

## Agentation development feedback

A separate Chromium check passed against the development frontend on port 5173, using the Docker backend on port 8080:

- Activated **Start feedback mode**, selected the sign-in workflow card heading, and added a visual annotation.
- Copied feedback through the toolbar and verified that the clipboard included the note, the selected element’s CSS location, and the component source location (`src/pages/Auth.tsx`).
- Captured and visually reviewed the annotation screenshot; no uncaught browser errors were recorded.
- Confirmed the production frontend on port 8080 excludes the development feedback toolbar.

## Production frontend

TypeScript compilation and Vite production bundling passed. Fonts are bundled locally, pages are lazy-loaded, charts and shared dependencies have separate chunks, and direct icon imports keep the redesigned module graph at 741 modules, compared with 2,235 before optimization.

## Deployment and optional integrations

The root Docker image was built successfully as `analytiq:latest` using Docker Desktop's Linux engine. The `analytiq` container was started on port 8080 and passed its health check. The dashboard, API, and embedded ML subprocesses ran together in this one container as non-root user `analyst` (UID 10001).

The default one-service Compose configuration passed validation. An earlier restart check verified persistence of 2 users, 4 datasets, and 1 completed trained model, including the saved pipeline file. The redesigned application images were subsequently deployed by replacing the container while retaining the same `analytiq-data` volume. The final container was confirmed `running` and `healthy`, with user `analyst` and the volume mounted at `/app/data`.

The original multi-service PostgreSQL/Nginx deployment remains in `docker-compose.multi.yml`; that configuration was not executed during this verification. The optional OpenAI integration was exercised through its no-key fallback; live language-model calls require a configured API key.

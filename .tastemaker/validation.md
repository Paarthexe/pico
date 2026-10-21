# Initial validation

2026-10-21

- Full Streamlit demo interaction validated with AppTest, including results and stale settings.
- Generated stratified chart rendered and visually inspected; long quartile intervals replaced by Q1–Q4 axis labels. Exact bin bounds remain in evidence tables.
- Browser connection attempted; runtime reported no browser available and discovery returned an empty list. Browser viewport, responsive layout, and full accessibility review remain unverified.
- Extracted embedded CSS passed anti-slop, motion, and component coherence scans. The scanners do not accept Python files, so only actual extracted CSS was counted.
- Palette contrast contract recorded in style-lock.md.
- Project style lock and pending decision log added. No personal profile promotion.
- Streamlit native controls used; React registries are incompatible with this Python stack.

## Generated Python extension

- Reused the locked palette and native Streamlit controls, with styles extracted into ui/styles.css.
- AppTest covers engine selection, model/engine settings invalidation, and unavailable Docker errors.
- Added execution logs/source to evidence and failed attempts to the investigation trail.
- No new browser visual review or personal profile promotion is claimed.

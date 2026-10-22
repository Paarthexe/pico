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

## Browser review and accessibility checks

- Real isolated Chromium tested desktop at 1280×900 and responsive widths 320, 390, and 768px.
- Five browser tests passed, including an actual investigation, report ZIP download, mobile findings, descriptive chart alternatives, and keyboard-focus navigation.
- Axe WCAG 2 A/AA and 2.1 A/AA rules reported no violations in the tested stable states. This is scoped automated coverage, not full assistive-technology certification.
- Fixed measured contrast failures in native captions, uploader hint text, and alert text. Preserved the locked palette; captions now use opaque muted text.
- Desktop and mobile rendered screenshots were inspected. Charts retain numerical evidence tables for a readable alternative at narrow widths.
- Project style lock and pending-review decision log updated. Personal profile unchanged.

2026-10-22: Seven Chromium checks pass, including expanded study controls at 390/1280px and WCAG A/AA scans. Complete panels and host effect plot visually reviewed. Generic design scanners parse no Python files and are not coverage. Locked palette/type unchanged. No personal profile changes.

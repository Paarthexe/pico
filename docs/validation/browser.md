# Browser validation

Date: 2026-10-21. Browser: isolated Playwright Chromium, no user profile.

Five passing checks cover:

- Desktop 1280×900: CSV demo, question input, five executed catalog analyses, findings, report tab, ZIP download.
- Responsive 320×844, 390×844, 768×844: layout overflow, form focus/navigation, WCAG scans.
- Mobile 390×844: populated findings, descriptive chart alternatives, statistical review, overflow, WCAG scan.

Axe runs the WCAG 2 A/AA and 2.1 A/AA tags on stable initial and populated states. No violations remained in those scans. Initial failures identified inherited caption opacity, uploader hints, and alert-text contrast; all were fixed without disabling rules or excluding components.

Desktop and mobile screenshots were visually inspected. This is automated testing and screenshot review, not an exhaustive screen-reader or WCAG conformance certification. CI uploads the actual reports, screenshots, and downloaded evidence bundle as `browser-validation`.

Run: `PICO_PYTHON=.venv/bin/python npm run test:browser`.

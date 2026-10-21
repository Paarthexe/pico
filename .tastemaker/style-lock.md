# Pico style lock

Surface: Streamlit data workspace for analysts, Operate mode.
Thesis: let the dataset, evidence, and investigation trail carry the interface.
First viewport: sidebar upload/settings, concise title, real data health, investigation form.
Mood: technical, light. Density 7, motion 2, variance 3, art direction 4.
System: native Streamlit controls and Matplotlib charts; React registries do not apply.

## Palette and color contract
Generated with technical/light seed 71.
Text #121c22, background #f0f9fe, surface #e6eff4, primary #017ca8,
on-primary #ffffff, accent #516fb4, border #d6dfe4, muted text #4d5b64.
Caption opacity is 1; native caption/uploader hint alpha had failed browser contrast checks.
Alert text uses the locked ink color for contrast against light state fills.
Text/background 16.20:1, text/surface 14.83:1, white/primary 4.72:1.
Accent/background 4.60:1. Primary/background 4.42:1: UI only, no small text.
Hairline borders are decorative; focus uses primary with 4.42:1 against background.
Dark mode: one locked light mode, no runtime toggle.

## Type, space, interaction
IBM Plex Sans/system sans, mono only for code. System fallback works offline.
Native Streamlit controls retain their keyboard, loading, and error behavior.
6px control radius. Spacing uses 8/16/24/32/48px steps.
Motion: 120ms opacity feedback, disabled for reduced motion.
No gradients, invented stats, photos, decorative illustrations, or marketing card grid.
Risk: custom Streamlit CSS may need adjustment after framework upgrades.

# Thoth — Design

Status: concept locked, wireframes + polish spec locked, pre-implementation.
Designer: Cassian (Metalab, 15y). Captured: 2026-04-21.

---

## 0. The direction: The Ledger

A naturalist's field journal, digitized. Two-column layout: a chronological stream of detections on the left (intelligently collapsed by species), a cinematic detail pane on the right. Day-shaped by default. Scales from a 20-second glance to a 20-minute deep dive without a mode switch.

Why this direction over the alternatives (The Window, The Constellation):

- **The Window** (ambient, one hero frame) is seductive but thin — after a week you've seen the trick.
- **The Constellation** (24-hour radial visualization) is the most designerly answer, but visualization-first UIs make lousy daily drivers. Becomes an earned "Patterns" tab later, not the front door.
- **The Ledger** has the spine a homelabber lives with. Extends cleanly to skywatch and skylapse later — more rows in more ledgers, sharing one detail pane.

The move: build the Ledger, make the detail pane *cinematic* (where The Window's warmth lives inside the Ledger's bones).

---

## 1. Layout grid

Two columns. Left is the stream. Right is the pane. Nothing else competes.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Thoth · Mon Apr 21 · [‹ Apr 20  Apr 21 ›]   [live●]  [⌕]  [filter]  │ 56px header
├────────────────────────┬─────────────────────────────────────────────┤
│                        │                                             │
│   STREAM               │   DETAIL PANE                               │
│   (scrollable)         │   (sticky, doesn't scroll with stream)      │
│                        │                                             │
│   380–420px            │   fluid, fills rest                         │
│                        │                                             │
└────────────────────────┴─────────────────────────────────────────────┘
```

- **Laptop (14–16")**: stream 400px fixed, pane fluid. Max page width 1600px, centered.
- **Wall tablet (10" landscape)**: stream 340px, pane fluid. Header shrinks to 48px. Touch targets min 44px.
- **Below ~900px wide**: stream goes full-width, tapping a row pushes pane as a full-screen overlay with back chevron. Don't design for phone first — this is a wall display and a desk app.

Gutter between columns: 1px hairline divider, not a gap. The stream and pane feel like one instrument.

---

## 2. The stream (left column)

### Collapsed row (default)

```
┌────────────────────────────────────────┐
│ ▸ [thumb] Black-capped Chickadee    ×7 │
│          Poecile atricapillus          │
│          4:12 – 4:21 PM  · feeder · 94%│
└────────────────────────────────────────┘
```

- Thumb: 48×48, rounded 6px, best-frame from the group.
- Common name: 15px, medium weight, primary text.
- Latin: 12px, italic, **opacity 0.56** (not 0.6 — see §7). Baseline 6px below common name, not the standard 4px.
- Count badge `×7`: right-aligned, tabular numerals, only if >1.
- Meta row: 12px, 55% opacity. `time range · camera · avg confidence`.
- Chevron `▸` on hover/focus signals expandable.

### Expanded row

Chevron rotates, reveals a horizontal strip of up-to-8 thumbnails (56×56, scrollable if more). Click a thumbnail → that detection loads in the pane. Row stays selected-highlighted.

### Grouping rules

Aggressive — the whole point of the Ledger is cognitive relief:

- Same species.
- Gap ≤ **10 minutes** between detections.
- Break on: 10+ min gap, day boundary, or confidence crossing the low-confidence threshold (a 45% chickadee and a 96% chickadee don't belong in the same row).
- Camera switch does **not** break a group. Chickadee on feeder cam + wide cam two minutes apart is one event.

### "Now" marker

When live, a thin 1px accent line pulses at the top of the stream with a subtle `LIVE` label. New detections slide in above with a 200ms fade. No bounce.

### Day dividers

Sticky sub-header, 32px tall. `Mon Apr 21` left, `23 detections · 11 species` right, 12px at 50% opacity. Sticks under the main header as you scroll.

---

## 3. The detail pane (right column)

Hero image dominates. ~60% of pane height on laptop, more on wall tablet.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│            [  HERO IMAGE 16:10  ]                │
│                                                  │
├──────────────────────────────────────────────────┤
│ [cam1▪][cam2 ][cam3 ][cam4 ]    seed feeder · IMX│
├──────────────────────────────────────────────────┤
│ ▶ ▁▂▅█▆▃▂▁▁▂▅▇█▇▅▂▁▁▁   0:04 / 0:12              │
├──────────────────────────────────────────────────┤
│ Black-capped Chickadee              4:17:23 PM   │
│ Poecile atricapillus                94% confident│
│ ─────────────────────────────────────────────    │
│ First seen today · 3rd visit · 48°F, light wind  │
└──────────────────────────────────────────────────┘
```

### Hero image

- `object-fit: contain` on a near-black matte — `oklch(0.18 0.01 90)`. Warm near-black, a hair toward browns in bird plumage. Black mattes make feathers look like cutouts; warm mattes make the bird feel like it's in the room.
- No drop shadows. No rounded corners on the image itself. Photos are the art; the chrome is the frame.

### Portrait-in-hero handling

If the source is portrait, do NOT letterbox. Blur-extend the edges (gaussian σ=40px, 30% matte overlay) so the subject sits centered in its native aspect, floating on a soft extension of itself. Apple Photos move — almost no dashboards bother.

### Thumbnail crops

4:5 portrait crops via a saliency pass (CLIP or small MobileNet head → bounding box → crop 4:5 around box, padded 12%). Center-crop fallback only when confidence <0.4 (~5% of the time).

### Bad frames

Compute Laplacian variance on ingest; below threshold gets a `low-quality` flag. Shown at 65% opacity in thumbs with a small `sketch` badge (not `blurry` — "blurry" sounds like a defect, "sketch" sounds intentional). Click-through shows full-size without the dim.

### Camera strip

4 small tabs directly under the hero. Active one filled, others outlined. If only 1 camera saw the bird, strip collapses to a single label (`seed feeder · IMX519`) — no empty slots. If 2–3 cameras, show only those. Never show disabled tabs; that's noise about capability, not content.

### Audio waveform

**The signature visual of the product. Default HTML5 `<audio>` is not on the table.**

- Render to canvas at 2× DPR. Bars 2px wide, 1px gaps, 72 bars per second. Peak-normalized per clip (not globally).
- Unplayed bars: `oklch(0.55 0.04 240 / 0.35)`. Played bars: accent color at full opacity.
- **The scrubber is not a line — it's the color boundary itself** between played and unplayed bars.
- **Detection window band**: soft vertical band behind the waveform, `oklch(from var(--accent) l c h / 0.12)`, 1px top/bottom border at 0.25 opacity. On play, the band pulses **once** (0.12 → 0.20 → 0.12 over 600ms). Do it once, not on loop — looping is a toy, once is a statement.
- Click-to-seek anywhere. Drag is frame-accurate. Space toggles play globally when a clip has focus. Waveform height 48px.
- Reference: Descript's waveform, minus the edit affordances.

### Metadata block

Two-column, type-driven. No icons, no cards. Common name bold, Latin italic under it. Time and confidence right-aligned on the same baselines. Hairline rule, then context line: first-seen today, visit count, weather snapshot. That context line is what makes this feel like a naturalist's field journal instead of a security camera.

**No action buttons by default.** No share, no download, no "mark as." If a "keep this one forever" pin becomes worth adding, it's one icon top-right of the hero.

---

## 4. States

- **Empty (no detections today)** — see §7, this is a craft opportunity, not a fallback.
- **Loading** — skeleton rows in the stream (3 of them, shimmer at 1.5s), pane shows matte background with hairline progress bar top. No spinners.
- **Low-confidence (<60%)** — row shows species name in italic + `?` (`Song Sparrow?`). Thumb has subtle dashed border. In the pane, confidence number is amber instead of neutral. Don't hide them — you want to see what BirdNET is uncertain about.
- **Simultaneous detections** — separate rows, newest-first. Don't merge a hummingbird and a chickadee because they happened in the same minute. The stream is a log, not a summary.
- **End of day** — stream ends: `— end of Apr 21 · 147 detections · 18 species —`. Below it, yesterday's first detection greyed to ~40%, inviting continued scroll into history.
- **Offline (Horus down)** — `live●` dot goes amber, label reads `last seen 4:32 PM`. Single banner under header: `Horus hasn't reported in 12 minutes.` No modal, no red alarm. This is a homelab, not a SOC.

---

## 5. Header / chrome

**Essential:**

- Date (today by default, 16px medium).
- 7-day calendar strip centered on current day, arrows for prev/next week. Click a day, jump.
- Live/paused dot. One click toggles auto-scroll-to-new. Paused means new detections collect above with a `↑ 4 new` pill.
- Search (`⌕`) — opens a **Linear-style command palette overlay**, searches species names and Latin. Not a sidebar, not a filter drawer.
- Filter — inline filter chips, always visible but subdued. Species multi-select, camera multi-select, confidence slider. **Not** a button that opens a panel.

**Defer:** Patterns tab, export, settings (tuck under a small gear bottom-left of stream later). Nothing else earns header real estate yet.

---

## 6. Typography + color

### Type

- **UI**: Inter Display (or Söhne family). Weight 550 for species common names, tracking -0.015em at 22px, -0.01em at 17px, 0 at 13px. Inter *Display* at display sizes, Inter regular below 20px. Optical sizing is not optional.
- **Timestamps / counts / confidence**: Söhne Mono or JetBrains Mono. `font-variant-numeric: tabular-nums` on root. Tabular numerals in a list is the single most common tell that nobody cared.
- **Optional serif** (Söhne Breit, Tiempos, or Source Serif) for species common names only. One concession to "naturalist" — gives the bird names weight without turning the app into a Ken Burns documentary.
- **Hanging punctuation**: `hanging-punctuation: first last`. Supported in Safari, progressively enhances elsewhere. Costs nothing. Nobody else does it.
- **Ligatures** on, discretionary ligatures off. `font-feature-settings: "liga" 1, "dlig" 0`.

### Color

- **Dark mode primary**, light mode secondary. Wall display in a home — dark wins 90% of the day.
- Background near-black `#0c0c0e` (for image matte use the warm variant `oklch(0.18 0.01 90)`).
- Surface `#17181b`. Text `#e8e8ea` / `#9a9aa0` / `#606066`.
- **One accent** — warm ochre `#c89b5c`. Used for: live dot, selected state, confidence high-water.
- **No bird-family color coding.** The photos are the color. If every Turdidae row is orange-tinted, the app looks like a taxonomy lesson instead of a window.

---

## 7. Motion, micro-craft, and the moments that make it Metalab-grade

### Motion discipline

- **One easing curve**: `cubic-bezier(0.32, 0.72, 0, 1)` (the curve Vercel and Linear both converged on — decelerating but not floaty). Use for 95% of everything.
- **Two durations**: 240ms for enter/exit of new content, 120ms for hover/micro-feedback. Do not invent a third.
- **New detection arrival**: row slides down from y=-8px with opacity 0→1 over 280ms, staggered 40ms behind a soft 180ms background flash of `oklch(from var(--accent) l c h / 0.08)` on the row. The flash decays; the row stays. Don't skip the flash — it's the "something just happened" moment.
- **Row-click cross-fade**: outgoing at scale(0.98) + opacity 0 over 160ms; incoming at scale(1.02) → 1.0 over 240ms. The overshoot is the polish. If the hero image snaps, the whole thing feels like a CCTV viewer. If it breathes, it feels like turning a page. **Spend the afternoon on this.**
- **What does NOT animate**: scrolling, text, anything users initiate directly. Animation on scroll is amateur hour.
- **Respect `prefers-reduced-motion`** — collapse all durations to 0.01ms (not 0; browsers treat 0 inconsistently).

### Latin name pair

Most-repeated typographic unit in the product. If ordinary, everything reads ordinary.

- Common name on top, Latin italic directly under, opacity **0.56** (not 0.6), baseline 6px below common (not 4px leading). The extra breath is the whole effect.

### Empty state — the "Quiet so far. Listening." moment

Treat as a feature, not a fallback. This is the screenshot-worthy moment.

- Centered. `Quiet so far.` at 22px/550. `Listening.` new line, 17px/400, opacity 0.56.
- Below: 1px hairline, 120px wide, `oklch(0.55 0.04 240 / 0.15)`. Tiny pulsing dot at its center — 6px, accent color, opacity 0.4 → 1.0 → 0.4 over 2.4s on the same easing. One breath every 2.4 seconds. **That's the station's heartbeat.**
- When a detection finally arrives at 3am: dot freezes mid-pulse at full opacity for 200ms, line extends to full width over 400ms, whole empty state cross-fades out as the detection row slides in from above. The user feels the station *finding* something.

### Two "oh" moments

1. **Live count in tab title.** `(3) Ledger — Thoth` updates live, 4-second debounce, favicon badge only when tab is unfocused. When you Cmd-Tab back, badge fades out over 600ms as tab regains focus. Makes the app feel like it was running while you were gone.
2. **Timestamp hover toggle.** Timestamps read `14m ago` by default. Hover → 100ms cross-fade to `3:47 PM`. No tooltip, no click — the text itself transforms in place. Stolen from GitHub, refined. Answers a question you had without feeling like a gimmick.

### Keyboard + focus

- `j/k` next/previous detection. `space` play/pause focused clip. `/` command palette. `g then t` go to today. `?` shortcut sheet.
- Focus rings: `outline: 2px solid oklch(0.72 0.15 240); outline-offset: 3px;`. Never browser default. Never box-shadow ring (clips on rounded corners).
- `:focus-visible`, not `:focus` — keyboard shows focus, mouse doesn't.

### Things to remove

- **Row dividers.** Delete them. Use 20px vertical spacing and let photo edges do the work. Dividers are what you add when the spacing isn't doing its job.
- **Secondary thumbnail metadata** (camera ID, duration badge, confidence pill all at once). Pick one — confidence — and only show when <0.75. Everything else moves to expanded detail.
- **The station-health strip.** Demote to single 8px dot in the header next to the logo. Green/amber/red. Click opens detail. 95% of the time everything is green and the strip is visual debt.

---

## 8. If you only do three things

Everything in §7 compounds off these three. Skip them and the rest is decoration.

1. **Fix the species-name typography pair.** Latin name at 0.56 opacity, italic, 6px baseline breath above. Most-repeated unit; sets the tone for everything.
2. **Build the waveform properly, including the pulse-once detection band.** Single most distinctive visual in the product; the one users will screenshot.
3. **Nail the empty state and the transition out of it** when the first detection arrives. The moment that earns the product its personality.

---

## 9. Don't overthink

The exact 10-minute grouping window. You will want to make it configurable, per-species, adaptive, learned. Don't. Ship 10 minutes flat, live with it for a month, tune one number later. Bird behavior is not your backend problem yet.

---

## 10. Data constraints (for implementation)

- Audio detections from BirdNET-Go on Horus (species, confidence, timestamp, audio clip).
- Image detections from 4 cameras on Horus: IMX519 feeder (close-up perched), Owlsight wide (yard context), IMX462 hummingbird (fast/low-light), night cam (owls).
- **High-quality stills prioritized** over video on the feeder cam.
- Retention: 30 days full-res images, thumbnails forever, metadata forever.

---

*Captured from Cassian's wireframe pass + Metalab-quality polish pass, 2026-04-21.*

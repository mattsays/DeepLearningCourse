// Shared definitions for the report: the TODO marker, the "not run" marker and the
// diagram blocks used by the architecture figures.
//
// Imported by report.typ and, so that a section file stays editable on its own, by
// every file under sections/.

// ---------------------------------------------------------------- markers
#let todo(body) = box(
  fill: rgb("#fde8e8"),
  stroke: 0.5pt + rgb("#c0392b"),
  inset: (x: 3pt, y: 1.5pt),
  radius: 2pt,
  text(fill: rgb("#c0392b"), size: 0.85em)[*TODO:* #body],
)
#let notrun = text(fill: luma(110), style: "italic")[not run]

// diagram blocks: frozen = grey, trained = blue, parameter-free = white
#let frozen-fill = luma(225)
#let train-fill = rgb("#d6e6f7")
#let blk(body, fill: white) = box(
  fill: fill,
  stroke: 0.5pt + luma(90),
  inset: (x: 4pt, y: 3pt),
  radius: 2pt,
  text(size: 7.5pt, body),
)
#let arr = text(size: 8pt, fill: luma(80))[→]

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


// ---------------------------------------------------------------- charts
// Horizontal grouped bar chart drawn with native primitives (no package download needed).
//   rows:   array of (label, (v1, v2, ...)), one value per series; `none` skips a bar
//   series: array of (name, colour), in the same order as the values of each row
//   refs:   array of (value, label), drawn as dashed vertical reference lines
// Values outside [lo, hi] are clamped for drawing; the printed number is always the true one.
#let hbar-chart(rows, series: (), refs: (), lo: 50, hi: 85, ticks: (50, 60, 70, 80),
  label-w: 3.8cm, plot-w: 8cm) = {
  let bh = 5pt
  let gap = 1.2pt
  let rgap = 5pt
  let head = 9pt
  let n = series.len()
  let row-h = n * bh + (n - 1) * gap
  let plot-h = rows.len() * (row-h + rgap) - rgap
  let x(v) = (calc.min(calc.max(v, lo), hi) - lo) / (hi - lo) * plot-w
  let axis-y = head + plot-h + 3pt
  box(width: label-w + plot-w + 0.8cm, height: axis-y + 20pt, {
    for t in ticks {
      place(dx: label-w + x(t), dy: head - 2pt,
        line(start: (0pt, 0pt), end: (0pt, plot-h + 5pt), stroke: 0.3pt + luma(200)))
      place(dx: label-w + x(t) - 8pt, dy: axis-y + 2pt,
        box(width: 16pt, align(center, text(size: 6.5pt)[#t])))
    }
    place(dx: label-w, dy: axis-y, line(length: plot-w, stroke: 0.5pt + luma(90)))
    for (v, name) in refs {
      place(dx: label-w + x(v), dy: head - 2pt,
        line(start: (0pt, 0pt), end: (0pt, plot-h + 5pt),
          stroke: (paint: rgb("#c0392b"), thickness: 0.6pt, dash: "dashed")))
      place(dx: label-w + x(v) + 2pt, dy: 0pt, text(size: 6pt, fill: rgb("#c0392b"))[#name])
    }
    for (i, row) in rows.enumerate() {
      let (label, vals) = row
      let y0 = head + i * (row-h + rgap)
      place(dx: 0pt, dy: y0 + row-h / 2 - 4pt,
        box(width: label-w - 6pt, align(right, text(size: 7.5pt, label))))
      for (j, v) in vals.enumerate() {
        if v != none {
          let y = y0 + j * (bh + gap)
          place(dx: label-w, dy: y,
            rect(width: x(v), height: bh, fill: series.at(j).at(1), stroke: none))
          place(dx: label-w + x(v) + 1.5pt, dy: y - 1.5pt,
            box(fill: white.transparentize(15%), inset: (x: 1pt, y: 0.5pt),
              text(size: 6pt)[#v]))
        }
      }
    }
    for (j, s) in series.enumerate() {
      place(dx: label-w + j * 3.4cm, dy: axis-y + 12pt, {
        box(rect(width: 7pt, height: 5pt, fill: s.at(1), stroke: none))
        h(3pt)
        text(size: 6.5pt, s.at(0))
      })
    }
  })
}

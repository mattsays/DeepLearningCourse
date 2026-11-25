// Multimodal Deception Detection on DOLOS — course report.
// Compile: typst compile report/report.typ   (or paste report.typ and the whole
// report/ folder into typst.app together with refs.bib)
//
// This file is the entry point only: it holds the page and style setup and pulls in
// the sections. The text lives in report/sections/:
//
//   00-title.typ              title block and abstract (not numbered)
//   01-introduction.typ       §1
//   02-related-work.typ       §2
//   03-data.typ               §3  dataset and evaluation protocol
//   04-baseline.typ           §4  theory-driven baseline
//   05-architectures.typ      §5  index only: one file per architecture, in 05-architectures/
//   06-best-model.typ         §6  analysis of Cross-Modal Fusion
//   07-discussion.typ         §7
//   08-conclusion.typ         §8
//   appendix/a-subject-independent.typ   Appendix A
//   appendix/b-implementation.typ        Appendix B
//
// Shared helpers (the TODO box, the "not run" marker and the diagram blocks) are in
// report/template.typ. Section numbering, cross-references and the bibliography index
// are computed over the whole document, so splitting a file never changes a number.
//
// No TODO box is left: every number is taken from a run's results file. Do not add a
// remembered number.
//
// Sources of the numbers (paths relative to notebooks/architectures/ of the code tree):
//   Cross-Modal Fusion   cross-modal-fusion/results/{late,video,audio}-{official,subject}/
//   VideoMAE + HuBERT    lore/results/fusion-video-lora-r8-official+audio-lora-r8-official/
//                        lore/results/fusion-video-lora-r8+audio-lora-r8/
//   Theory baseline      load/results/load-{official,subject}/
//   Temporal CNN         teo/results/temporal-{official,subject,official-video,official-audio}/
//   MARLIN + ConvNeXt    fab/results/marlin-{official,subject}/preds.csv, rescored against
//                        dataset.csv (its summary.json / folds.csv are not reproduced)

#import "template.typ": todo, notrun, blk, arr, frozen-fill, train-fill

// ---------------------------------------------------------------- document setup
#set document(title: "Multimodal Deception Detection on the DOLOS Dataset")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)
#set figure(gap: 0.7em)
#show figure.caption: set text(size: 0.9em)
#set table(stroke: none, inset: (x: 5pt, y: 3pt))
#show table.cell.where(y: 0): set text(weight: "bold")
#show table: set text(size: 9pt)
#show table: set par(justify: false)
#set math.equation(numbering: "(1)")

// ---------------------------------------------------------------- front matter and body
#include "sections/00-title.typ"
#include "sections/01-introduction.typ"
#include "sections/02-related-work.typ"
#include "sections/03-data.typ"
#include "sections/04-baseline.typ"
#include "sections/05-architectures.typ"
#include "sections/06-best-model.typ"
#include "sections/07-discussion.typ"
#include "sections/08-conclusion.typ"

#bibliography("refs.bib", style: "ieee")

// ---------------------------------------------------------------- appendix
// Appendix headings restart the counter and switch to A.1 numbering; the set rule
// therefore belongs here, before the first appendix file, and not in a section file.
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

#include "sections/appendix/a-subject-independent.typ"
#include "sections/appendix/b-implementation.typ"

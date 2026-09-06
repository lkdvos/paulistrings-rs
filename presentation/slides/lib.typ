// Hand-rolled slide theme for the paulistrings-rs talk. No external packages.
// Palette mirrors examples/common/report.py so slide text and figures agree.

#let c-blue   = rgb("#2a78d6")
#let c-orange = rgb("#eb6834")
#let c-aqua   = rgb("#1baf7a")
#let c-yellow = rgb("#eda100")
#let c-red    = rgb("#e34948")
#let c-violet = rgb("#4a3aa7")
#let c-ink    = rgb("#1e1e1e")
#let c-muted  = rgb("#898781")
#let c-grid   = rgb("#e1e0d9")
#let c-paper  = rgb("#fbfaf7")
#let c-dark   = rgb("#17233a")

#let proverb = [_Wie niet sterk is, moet slim zijn._]

#let slide-counter = counter("slide")

#let footer-bar(dark: false) = {
  let fg = if dark { white.transparentize(35%) } else { c-muted }
  place(bottom, block(width: 100%, {
    set text(size: 9pt, fill: fg)
    grid(columns: (1fr, auto), align: (left, right),
      proverb,
      context slide-counter.display(),
    )
  }))
}

#let setup(body) = {
  set page(paper: "presentation-16-9", margin: (x: 1.5cm, top: 1.2cm, bottom: 1.1cm), fill: c-paper)
  set text(font: "Carlito", size: 18pt, fill: c-ink)
  show math.equation: set text(font: "New Computer Modern Math", size: 18pt)
  show raw: set text(font: "Source Code Pro", size: 14pt)
  set list(marker: text(fill: c-blue)[•], spacing: 0.6em)
  set enum(spacing: 0.75em)
  show strong: set text(fill: c-dark)
  body
}

// A regular content slide: title, thin accent rule, body, footer.
#let slide(title, body, note: none) = {
  slide-counter.step()
  page({
    block(width: 100%, below: 0.5em, {
      text(size: 26pt, weight: "bold", fill: c-dark, title)
      v(0.25em)
      line(length: 100%, stroke: 1.5pt + c-blue)
    })
    body
    if note != none {
      place(bottom + right, dy: -1.1em, text(size: 10pt, fill: c-muted, note))
    }
    footer-bar()
  })
}

// Section divider on a dark ground.
#let section(title, sub: none, act: none) = {
  slide-counter.step()
  set page(fill: c-dark, margin: (x: 1.5cm, y: 1.5cm))
  page({
    set text(fill: white)
    v(1fr)
    if act != none { text(size: 18pt, fill: c-aqua, weight: "bold", upper(act)); v(0.4em) }
    text(size: 44pt, weight: "bold", title)
    if sub != none { v(0.6em); text(size: 22pt, fill: white.transparentize(25%), sub) }
    v(1.4fr)
    footer-bar(dark: true)
  })
}

#let title-slide(title, sub, author, venue) = {
  slide-counter.step()
  set page(fill: c-dark, margin: (x: 1.5cm, y: 1.5cm))
  page({
    set text(fill: white)
    v(1fr)
    text(size: 24pt, fill: c-aqua, style: "italic", proverb)
    v(0.2em)
    text(size: 15pt, fill: white.transparentize(35%))["Those who are not strong must be smart." — Dutch proverb]
    v(1em)
    text(size: 40pt, weight: "bold", title)
    v(0.4em)
    text(size: 22pt, fill: white.transparentize(20%), sub)
    v(1.6fr)
    grid(columns: (1fr, auto), align: (left, right),
      text(size: 16pt, author), text(size: 16pt, fill: white.transparentize(35%), venue))
  })
}

// Figure with a small muted caption. `path` is relative to the repo root (compile with --root).
#let fig(path, caption: none, width: 100%, height: none) = {
  align(center, {
    if height != none { image(path, height: height) } else { image(path, width: width) }
    if caption != none { v(0.2em); text(size: 11pt, fill: c-muted, caption) }
  })
}

// Grey placeholder for a figure that is not produced yet.
#let placeholder(label, width: 100%, height: 9cm) = {
  align(center, rect(width: width, height: height, fill: c-grid, stroke: (paint: c-muted, dash: "dashed"),
    align(center + horizon, text(fill: c-muted, size: 14pt, label))))
}

#let two-col(left, right, ratio: (1fr, 1fr), gutter: 1.2cm) = {
  grid(columns: ratio, column-gutter: gutter, left, right)
}

// Callout box for the punchline of a slide.
#let punch(body, color: c-orange) = {
  block(width: 100%, inset: 0.6em, radius: 4pt, fill: color.lighten(88%), stroke: (left: 3pt + color),
    text(size: 17pt, body))
}

#let muted(body) = text(fill: c-muted, body)
#let small(body) = text(size: 14pt, body)
#let mono(body) = text(font: "Source Code Pro", size: 16pt, body)

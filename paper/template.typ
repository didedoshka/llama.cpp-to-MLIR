#let space() = {
  h(1.25cm)
}

#let generate_title() = {
  pagebreak(weak: true)

  align(
    center,
    par(leading: 0.65em)[
      *ПРАВИТЕЛЬСТВО РОССИЙСКОЙ ФЕДЕРАЦИИ* \
      #text(size: 11pt)[*ФГАОУ ВО*] *НАЦИОНАЛЬНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ УНИВЕРСИТЕТ \
      «ВЫСШАЯ ШКОЛА ЭКОНОМИКИ»*],
  )

  v(0.5em)

  align(
    center,
    par(leading: 0.5em)[
      Факультет компьютерных наук \
      Образовательная программа «Прикладная математика и информатика»],
  )

  v(7em)

  align(center)[
    #text(size: 14pt)[*Отчет об исследовательском проекте*] \
    #par(
      leading: 0.5em,
      text(
        size: 13pt,
      )[на тему "Компиляторная оптимизация вычислительных графов \ для локального выполнения больших языковых моделей"],
    )
  ]
  v(2em)
  [*Выполнил студент:* #v(2mm)]
  [
    #table(
      columns: (auto, auto),
      stroke: none,
      column-gutter: 1.5cm,
      [#h(1.3cm)группы \#БПМИ236, 3 курса], [Горохов Дмитрий Александрович],
    )
  ]
  v(1em)
  [#space() *Принял руководитель проекта:* #v(2mm)]
  [
    #table(
      columns: auto,
      stroke: none,
      [#h(1.3cm)Кулагин Иван Иванович, к. т. н.], [#h(1.3cm)Научный сотрудник], par(
        leading: 0.5em,
      )[#h(1.3cm)Федеральное государственное бюджетное учреждение науки \ #h(1.3cm)Институт системного программирования им. В.П. Иванникова \ #h(1.3cm)Российской академии наук (ИСП РАН)],
    )
  ]
  align(center + bottom)[
    Москва 2025
  ]
  pagebreak(weak: true)
}


#let init_font(doc) = {
  set text(
    font: "Times New Roman",
    size: 12pt,
    lang: "ru",
    // top-edge: "ascender",
    // bottom-edge: "descender",
  )
  set par(
    leading: 1.2em,
    spacing: 2em,
    first-line-indent: 1.25cm,
    justify: true,
  )

  doc
}



#let init_page(doc) = {
  let page_numbering(page_number) = {
    if page_number != 1 {
      page_number
    }
  }
  set page(
    paper: "a4",
    margin: (top: 2cm, left: 2.5cm, right: 1cm, bottom: 2cm),
    numbering: (x, ..) => page_numbering(x),
  )
  doc
}

#let init_headings(doc) = {
  show <no_numbering>: set heading(numbering: none)
  show <no_outline>: set heading(outlined: false)
  set heading(numbering: "1.1.1.1.1")
  show heading: it => {
    it
    v(1em)
  }
  doc
}

#let init_figures(doc) = {
  set figure(numbering: "1.1.1.1.1")
  show figure.where(kind: table): set figure.caption(position: top)
  doc
}

#let init_references(doc) = {
  set ref(supplement: none)
  doc
}

#let init_equations(doc) = {
  set math.equation(numbering: "(1.1.1.1.1)")
  show math.equation: set block(breakable: true)
  doc
}

#let term_paper(doc) = {
  show: init_font
  show: init_page
  show: init_headings
  show: init_figures
  show: init_references
  show: init_equations
  doc
}


#let generate_outline() = {
  pagebreak(weak: true)
  show outline.entry.where(level: 1): it => {
    v(12pt, weak: true)
    strong(it)
  }
  outline(indent: auto)
  pagebreak(weak: true)
}


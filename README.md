# ML Research Notebook

A [Quarto](https://quarto.org/) website for documenting ML research.

## Quickstart

```bash
quarto preview    # live-reload dev server
quarto render     # build to _site/
```

## Structure

```
research-notebook/
├── _quarto.yml
├── index.qmd
├── about.qmd
├── styles.css
├── new.sh                          # helper script to create entries
├── ideas/
│   ├── index.qmd                   # table listing
│   ├── _template.qmd               # template for new ideas
│   └── YYYY-MM-DD-slug/index.qmd
├── paper-notes/
│   ├── index.qmd                   # table listing
│   ├── _template.qmd               # template for paper notes
│   └── YYYY-MM-DD-slug/index.qmd
└── notes/
    ├── index.qmd                   # table listing
    ├── _template.qmd               # template for general notes
    └── YYYY-MM-DD-slug/index.qmd
```

## Adding a New Entry

```bash
# Use the helper script (auto-fills today's date)
./new.sh ideas dense-reward-shaping
./new.sh paper-notes attention-is-all-you-need
./new.sh notes gpu-profiling-results

# Then edit the created file
```

## Deploying

```bash
quarto publish gh-pages
quarto publish netlify
```

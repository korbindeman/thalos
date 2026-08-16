# Report source opened instead of embedded page

## Symptom

A visual summary displayed literal image placeholders such as
`{{img:path|caption}}` in the styled report instead of the captures. The image
publisher had succeeded and a self-contained report existed beside the file
the agent opened.

## Mechanism

The original workflow kept `report.html` as the editable token source and wrote
`report.embedded.html`; a first repair renamed the source to
`report.source.html` and the result to `report.html`. Both designs still left
two browser-renderable HTML files beside each other and relied on an instruction
to choose the right one. An agent opened `report.source.html`, so the browser
correctly rendered its unresolved token text. The embedder was not the failing
component; the source/result boundary was unenforced.

## Fix and recurrence tell

The editable input is now `report.html.in`, and its real report body is inside a
non-rendering source wrapper. Opening the input shows only an **Unpublished
report input** warning. The one agent entry point, `just publish-report`, accepts
only that suffix and wrapper, embeds every image, rejects
zero/malformed/unresolved tokens and live image references, and prints the only
canonical `report.html` URL that may be opened. Regression tests cover the
publication failures and prove the wrapper is removed from the result.

Any `.source.html` or `.embedded.html` report, any literal image token in a
canonical `.html`, or any report image whose `src` is not a `data:image/` URI is
a recurrence. Do not repair it with stronger wording; restore the enforced
one-input/one-page boundary.

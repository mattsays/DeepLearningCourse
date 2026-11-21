#!/usr/bin/env bash
# Verify, or re-download, the papers indexed in references.md.
#
# The PDFs are committed; this script is the check that each file is the whole file (arXiv's
# byte count, not a truncated download) and the way back if one goes missing.
#
#   ./fetch_references.sh          # fetch what is missing
#   ./fetch_references.sh --check  # only report what is here
#
# Resumable and idempotent: a file whose byte count matches the server is left alone.
set -u
cd "$(dirname "$0")" || exit 1

PAPERS="
pecl-dolos             2303.12745
dinov2                 2304.07193
wavlm                  2110.13900
minilm                 2002.10957
vjepa2                 2506.09985
mult                   1906.00295
perceiver              2103.03206
gmu                    1702.01992
adapters               1902.00751
lora                   2106.09685
multimodal-ml-survey   1705.09406
shortcut-learning      2004.07780
"

size() { wc -c < "$1" | tr -d ' '; }
remote() { curl -sSIL -m 60 "https://arxiv.org/pdf/$1" | tr -d '\r' |
           grep -i '^content-length:' | tail -1 | awk '{print $2}'; }

target() {   # target <slug> <id>  -> the filename references.md links to
  case "$1" in
    pecl-dolos) echo "2303.12745v2.pdf" ;;
    *)          echo "$1-$2.pdf" ;;
  esac
}

wanted=0; have=0; fetch=0
while read -r slug id; do
  [ -z "${slug:-}" ] && continue
  out=$(target "$slug" "$id")
  wanted=$((wanted + 1))
  if [ -s "$out" ] && [ "$(size "$out")" = "$(remote "$id")" ]; then
    have=$((have + 1))
    printf 'have  %-38s %10s bytes\n' "$out" "$(size "$out")"
  else
    fetch=$((fetch + 1))
    printf 'need  %-38s %s\n' "$out" "https://arxiv.org/pdf/$id"
  fi
done <<< "$PAPERS"

echo
echo "$have of $wanted present, $fetch to download"
[ "${1:-}" = "--check" ] && exit 0

while read -r slug id; do
  [ -z "${slug:-}" ] && continue
  out=$(target "$slug" "$id")
  head -c 5 "$out" 2>/dev/null | grep -q '%PDF' &&
    [ "$(size "$out")" = "$(remote "$id")" ] && continue
  echo "downloading $out"
  curl -sS -L -m 1800 -C - -o "$out" "https://arxiv.org/pdf/$id"
  if [ "$(size "$out")" = "$(remote "$id")" ]; then
    echo "  ok  $(size "$out") bytes"
  else
    echo "  INCOMPLETE: re-run to resume" >&2
  fi
done <<< "$PAPERS"

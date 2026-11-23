#!/bin/bash
# Write a download script for the Kepler long-cadence lightcurves of every KOI
# in a catalog, from the MAST directory listing of each KIC.
#
# usage: scripts/make_kepler_download.sh CATALOG OUTFILE
#   CATALOG  planet catalog with a kic_id column, e.g.
#            reference/alderaan/kepler_merged_alderaan.csv
#   OUTFILE  download script to write, e.g.
#            ~/data/MAST_downloads/Kepler_KOI_curl.sh
#
# Run in the alderaan conda env (needs pandas). One curl process gets every
# listing over one connection (~4000 small requests, a few minutes). Exits 1,
# writing nothing, if any KIC listing fails. Run OUTFILE in the download
# directory to fetch the files; it skips files already on disk, so it also
# serves as a completeness check.
set -euo pipefail
[ $# -eq 2 ] || { sed -n '2,15p' "$0"; exit 2; }
catalog=$1
outfile=$2
base=https://archive.stsci.edu/missions/kepler/lightcurves
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

python -c "
import sys
import pandas as pd
kics = pd.read_csv(sys.argv[1], usecols=['kic_id']).kic_id
print('\n'.join(f'{k:09d}' for k in sorted(set(kics))))
" "$catalog" > "$tmp/kics.txt"
echo "$(wc -l < "$tmp/kics.txt") KICs in $catalog"

# one curl process for every listing; a STATUS line follows each listing
sed "s|^\(....\)\(.*\)|url = \"$base/\1/\1\2/\"|" "$tmp/kics.txt" > "$tmp/list.cfg"
curl -sS --fail -K "$tmp/list.cfg" -w '\nSTATUS %{http_code} %{url}\n' \
  > "$tmp/raw.txt" || true
grep '^STATUS' "$tmp/raw.txt" > "$tmp/status.txt" || true
if [ "$(grep -c '^STATUS 200 ' "$tmp/status.txt" || true)" -ne "$(wc -l < "$tmp/kics.txt")" ]; then
  echo "error: some KIC listings failed:" >&2
  grep -v '^STATUS 200 ' "$tmp/status.txt" >&2 || true
  exit 1
fi
grep -o 'kplr[0-9]\{9\}-[0-9]*_llc\.fits' "$tmp/raw.txt" | sort -u > "$tmp/files.txt"
nokic=$(comm -23 "$tmp/kics.txt" <(cut -c5-13 "$tmp/files.txt" | sort -u))
if [ -n "$nokic" ]; then
  echo "error: no long-cadence files listed for KIC(s):" $nokic >&2
  exit 1
fi
echo "$(wc -l < "$tmp/files.txt") long-cadence files listed"

{
cat <<HEAD
#!/bin/bash
# Kepler long-cadence lightcurves (*_llc.fits) for every KOI in
# $(basename "$catalog"), from the MAST archive. Written by
# scripts/make_kepler_download.sh (alderaan) on $(date -u +%Y-%m-%d).
#
# Run in the download directory: bash $(basename "$outfile")
# One curl process fetches every file over one connection. Files already on
# disk are skipped. Each file downloads as *.part and is renamed only after
# HTTP 200, so an interrupted run leaves no truncated file. Rerun to resume.
HEAD
cat <<'BODY'
base=https://archive.stsci.edu/missions/kepler/lightcurves
cfg=$(mktemp)
trap 'rm -f "$cfg"' EXIT
while read -r f; do
  [ -s "$f" ] && continue
  printf 'url = "%s/%s/%s/%s"\noutput = "%s.part"\n' \
    "$base" "${f:4:4}" "${f:4:9}" "$f" "$f"
done > "$cfg" <<'EOF'
BODY
cat "$tmp/files.txt"
cat <<'TAIL'
EOF
n=$(grep -c '^url' "$cfg" || true)
echo "$n file(s) to download"
[ "$n" -gt 0 ] || exit 0
ok=0
while read -r code part; do
  if [ "$code" = 200 ]; then
    mv "$part" "${part%.part}"
    ok=$((ok + 1))
  else
    echo "FAILED ($code): ${part%.part}" >&2
  fi
done < <(curl -sS --fail --remove-on-error -K "$cfg" \
  -w '%{http_code} %{filename_effective}\n')
echo "downloaded $ok/$n"
[ "$ok" -eq "$n" ]
TAIL
} > "$outfile"
chmod +x "$outfile"
echo "wrote $outfile"

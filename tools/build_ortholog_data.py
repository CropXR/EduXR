"""Generate docs/ortholog_data.js for the W4 translational-biology practical page.

The page needs to answer, for any of the 500 genes of the biotic-stress dataset,
"does this gene have an orthologue in a crop, and how similar is it?". The source
for that is Ensembl Plants Compara, which is far too large to ship to a browser:
the complete homology dump is 8.4 GB. This script prunes it to the 500 genes and
the ten species the page shows, which comes to a few hundred KB.

The gene list is read straight out of docs/go_data.js, which tools/build_go_data.py
in turn derived from docs/transcriptomics_clustering.html, so the Week 2 and
Week 4 pages can never disagree about which genes they are showing.

Two sources, both from the Ensembl Plants FTP. The REST API and BioMart would
both be tidier, but rest.ensembl.org returns HTTP 500 for every plants endpoint
and the BioMart service returns empty responses.

  1. the per-genome Compara homology dumps, which give one row per homologous
     gene pair with the one2one / one2many / many2many cardinality and the
     percent identity. Note that each dump holds an *arbitrary half* of the
     pairs involving that genome, so getting every Arabidopsis-to-rice orthologue
     means reading both the arabidopsis_thaliana dump and the oryza_sativa one.
     That is why this reads eleven files rather than one.
  2. the peptide FASTA of the species we align, which is the only place the
     protein sequences live. Protein ids there match the Compara
     protein_stable_id exactly, which is why this beats UniProt.

Those files come to roughly 700 MB, so they are never written to disk. Each one
is decompressed as it streams in, and only the handful of rows that survive the
filter is cached, under data/_ensembl_cache/ (gitignored). The first run takes a
couple of minutes; later runs read the cache and finish in seconds.

Run:

    python tools/build_ortholog_data.py
    python tools/build_ortholog_data.py --no-seq     # skip the sequence pass
    python tools/build_ortholog_data.py --refresh    # ignore the cache
"""

import gzip
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CACHE = ROOT / "data" / "_ensembl_cache"

FTP = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current"
RELEASE = "116"

# The species the page shows, in the order they appear in the table and the
# tree. Ten is deliberate: enough to make the point, few enough to read.
#   latin, common name, clade label, download its peptide FASTA?
SPECIES = [
    ("brassica_oleracea",    "Cabbage",     "Brassica, same family",     True),
    ("vitis_vinifera",       "Grapevine",   "Eudicot",                   True),
    ("solanum_tuberosum",    "Potato",      "Eudicot",                   True),
    ("glycine_max",          "Soybean",     "Eudicot, legume",           True),
    ("oryza_sativa",         "Rice",        "Grass",                     True),
    ("zea_mays",             "Maize",       "Grass",                     False),
    ("triticum_aestivum",    "Bread wheat", "Grass, six sets of chromosomes", True),
    ("hordeum_vulgare",      "Barley",      "Grass",                     False),
    ("amborella_trichopoda", "Amborella",   "Earliest flowering plant",  False),
    ("physcomitrium_patens", "Moss",        "Non-flowering plant",       False),
]
SPECIES_INDEX = {latin: i for i, (latin, *_r) in enumerate(SPECIES)}

# Ensembl's homology_type values, collapsed to the three cardinalities the page
# talks about. Paralogs and gene splits are counted separately.
TYPE_CODE = {
    "ortholog_one2one": 1,
    "ortholog_one2many": 2,
    "ortholog_many2many": 3,
}
PARALOG_TYPES = {"within_species_paralog", "other_paralog", "gene_split"}


CACHE_COLUMNS = ["at_gene", "at_protein", "species", "gene", "protein",
                 "type", "identity", "high_confidence"]


def log(msg):
    print(msg, flush=True)


def stream_lines(url):
    """Decompress a remote .gz as it arrives, without ever storing it."""
    with urllib.request.urlopen(url, timeout=180) as resp:
        with gzip.open(resp, "rt", errors="replace") as fh:
            yield from fh


def homology_url(latin):
    return (f"{FTP}/tsv/ensembl-compara/homologies/{latin}/"
            f"Compara.{RELEASE}.protein_default.homologies.tsv.gz")


_PEP_URL_CACHE = {}


def peptide_url(latin):
    """Find the *.pep.all.fa.gz filename, which carries the assembly version."""
    base = f"{FTP}/fasta/{latin}/pep/"
    with urllib.request.urlopen(base, timeout=60) as r:
        listing = r.read().decode("utf-8", "replace")
    names = re.findall(r'href="([^"]+\.pep\.all\.fa\.gz)"', listing)
    if not names:
        raise SystemExit(f"no peptide FASTA listed for {latin} at {base}")
    return base + names[0]


def read_gene_list():
    """The 500 gene ids and their TAIR symbols, from docs/go_data.js."""
    src = (DOCS / "go_data.js").read_text(encoding="utf-8")

    def const(name, opener):
        m = re.search(rf"const {name}\s*=\s*(\{opener}.*?);\n", src, re.S)
        if not m:
            raise SystemExit(f"docs/go_data.js: could not find const {name}")
        return json.loads(m.group(1))

    return const("GENES", "["), const("GO_SYMBOL", "{")


def scan_homologies(latin, wanted, refresh=False):
    """Rows of one genome's homology dump that pair our genes with our species.

    Rows are cached in normalised form, always with the Arabidopsis gene first,
    because the dump puts it on whichever side it likes.
    """
    cached = CACHE / f"homology_{latin}.tsv"
    if cached.exists() and not refresh:
        rows = [line.rstrip("\n").split("\t")
                for line in cached.read_text(encoding="utf-8").splitlines()[1:]]
        log(f"  cached   {latin}: {len(rows):,} rows")
        return rows

    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = stream_lines(homology_url(latin))
    col = {name: i for i, name in enumerate(next(lines).rstrip("\n").split("\t"))}
    rows, total = [], 0
    for line in lines:
        total += 1
        f = line.rstrip("\n").split("\t")
        code = TYPE_CODE.get(f[col["homology_type"]])
        if code is None:
            continue
        # Arabidopsis can be on either side of the pair; normalise so it is first.
        if f[col["species"]] == "arabidopsis_thaliana":
            a, b = "", "homology_"
        elif f[col["homology_species"]] == "arabidopsis_thaliana":
            a, b = "homology_", ""
        else:
            continue
        at_gene = f[col[f"{a}gene_stable_id"]]
        other_sp = f[col[f"{b}species"]]
        if at_gene not in wanted or other_sp not in SPECIES_INDEX:
            continue
        rows.append([
            at_gene,
            f[col[f"{a}protein_stable_id"]],
            other_sp,
            f[col[f"{b}gene_stable_id"]],
            f[col[f"{b}protein_stable_id"]],
            str(code),
            str(round(float(f[col[f"{a}identity"]]))),
            f[col["is_high_confidence"]] or "0",
        ])

    cached.write_text(
        "\t".join(CACHE_COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in rows),
        encoding="utf-8")
    log(f"  streamed {latin}: {total:,} rows read in {time.time() - t0:.0f}s, "
        f"kept {len(rows):,}")
    return rows


def count_paralogs(wanted, refresh=False):
    """How many other Arabidopsis genes each gene is a paralog of.

    Only the arabidopsis_thaliana dump holds within-species pairs, and
    scan_homologies throws those rows away, so this makes a second pass over
    that one file. Both passes are cached, so it costs one extra scan on the
    first run and nothing afterwards.
    """
    cached = CACHE / "paralogs.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))

    counts = {}
    lines = stream_lines(homology_url("arabidopsis_thaliana"))
    col = {name: i for i, name in enumerate(next(lines).rstrip("\n").split("\t"))}
    for line in lines:
        f = line.rstrip("\n").split("\t")
        if f[col["homology_type"]] not in PARALOG_TYPES:
            continue
        gene = f[col["gene_stable_id"]]
        if gene in wanted:
            counts[gene] = counts.get(gene, 0) + 1
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(counts), encoding="utf-8")
    return counts


def collect(genes, refresh=False):
    """Every ortholog of our genes in our species, from both sides of each pair."""
    wanted = set(genes)
    seen, orth, protein_of = set(), {}, {}
    for latin in ["arabidopsis_thaliana"] + [s[0] for s in SPECIES]:
        for at_gene, at_prot, sp, gene, prot, code, ident, hc in \
                scan_homologies(latin, wanted, refresh):
            if (at_gene, gene) in seen:
                continue
            seen.add((at_gene, gene))
            orth.setdefault(at_gene, []).append(
                [SPECIES_INDEX[sp], gene, int(code), int(ident), int(hc)])
            protein_of[(at_gene, gene)] = (at_prot, prot)
    for rows in orth.values():
        rows.sort(key=lambda r: (r[0], -r[3]))
    return orth, protein_of


def alignment_partners(orth):
    """For each gene, the closest orthologue in each species we have sequences for.

    The alignment step follows whichever gene the student picked in step 3, so
    every gene needs sequences rather than a featured handful. Shipping every
    orthologue would run to several MB, so we ship the closest match per species:
    that is the one worth aligning, and the table in step 3 still lists the rest
    with their identities.
    """
    seq_species = {SPECIES_INDEX[l] for l, _c, _cl, has in SPECIES if has}
    picks = {}
    for gene, rows in orth.items():
        best = {}
        for sp, other, _type, ident, _hc in rows:
            if sp in seq_species and (sp not in best or ident > best[sp][1]):
                best[sp] = (other, ident)
        if best:
            picks[gene] = {sp: other for sp, (other, _i) in best.items()}
    return picks


def read_sequences(picks, protein_of, refresh=False):
    """Protein sequences for every gene and its closest orthologues, by gene id."""
    need = {}   # species latin -> {protein id -> gene id}
    for gene, by_species in picks.items():
        for sp_i, other in by_species.items():
            at_prot, other_prot = protein_of[(gene, other)]
            need.setdefault("arabidopsis_thaliana", {})[at_prot] = gene
            need.setdefault(SPECIES[sp_i][0], {})[other_prot] = other

    seqs = {}
    for latin, by_protein in sorted(need.items()):
        cached = CACHE / f"pep_{latin}.json"
        if cached.exists() and not refresh:
            found = json.loads(cached.read_text(encoding="utf-8"))
            if set(by_protein.values()) <= set(found):
                log(f"  cached   {latin}: {len(found)} sequences")
                seqs.update(found)
                continue
        t0 = time.time()
        found, keep, buf = {}, None, []
        for line in stream_lines(peptide_url(latin)):
            if line.startswith(">"):
                if keep:
                    found[keep] = "".join(buf)
                pid = line[1:].split(None, 1)[0]
                gene = by_protein.get(pid)
                keep = gene if gene and gene not in found else None
                buf = []
            elif keep:
                buf.append(line.strip())
        if keep:
            found[keep] = "".join(buf)
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(found), encoding="utf-8")
        log(f"  streamed {latin}: {len(found)}/{len(by_protein)} sequences "
            f"in {time.time() - t0:.0f}s")
        seqs.update(found)
    return seqs


_PEP_URL_CACHE = {}


# The species phylogeny is well established and is not something you would infer
# from a homology table, so it is written out by hand. Leaves index SPECIES;
# -1 is Arabidopsis itself.
TREE = ["Land plants", [
    ["Moss", [], 9],
    ["Vascular plants", [
        ["Amborella", [], 8],
        ["Grasses", [
            ["Rice", [], 4],
            ["Maize", [], 5],
            ["Wheat and barley", [
                ["Bread wheat", [], 6],
                ["Barley", [], 7],
            ]],
        ]],
        ["Eudicots", [
            ["Grapevine", [], 1],
            ["Potato", [], 2],
            ["Soybean", [], 3],
            ["Mustard family", [
                ["Arabidopsis", [], -1],
                ["Cabbage", [], 0],
            ]],
        ]],
    ]],
]]

HEADER = """/* Generated by tools/build_ortholog_data.py -- do not edit by hand.

   Ensembl Plants Compara orthologues of the 500 genes of the biotic stress
   dataset (Howard et al. 2013), pruned out of the per-genome homology dumps
   and the peptide FASTA files on the Ensembl Plants FTP.

   ORTH_SPECIES   [latin name, common name, clade label, has sequences]
   ORTH_TREE      species phylogeny; leaves index ORTH_SPECIES, -1 is Arabidopsis
   ORTH           gene -> [[species, ortholog gene id, type, identity %, high confidence], ...]
                  type 1 = one-to-one, 2 = one-to-many, 3 = many-to-many
                  identity is the percentage of the Arabidopsis protein that is
                  identical in the orthologue
   ORTH_PARALOGS  gene -> number of paralogs within Arabidopsis itself
   ORTH_SEQ       gene id -> protein sequence. Covers every Arabidopsis gene and,
                  for each, its closest orthologue in each of the species whose
                  peptide FASTA we read, which is what the alignment step needs.
*/

"""


def js_const(name, value, indent=None):
    return f"const {name} = {json.dumps(value, separators=(',', ':'), indent=indent)};\n\n"


def build(with_sequences=True, refresh=False):
    t0 = time.time()
    genes, symbols = read_gene_list()
    log(f"{len(genes)} genes read from docs/go_data.js")

    log("homologies (11 dumps, each holds an arbitrary half of the pairs):")
    orth, protein_of = collect(genes, refresh)
    paralogs = count_paralogs(set(genes), refresh)

    picks = alignment_partners(orth)
    seqs = {}
    if with_sequences:
        log("sequences:")
        seqs = read_sequences(picks, protein_of, refresh)
    alignable = sum(1 for g in genes if seqs.get(g)
                    and any(seqs.get(o) for o in picks.get(g, {}).values()))
    log(f"{len(seqs)} protein sequences kept; "
        f"{alignable}/{len(genes)} genes can be aligned against at least one orthologue")

    with_orth = sum(1 for g in genes if orth.get(g))
    log(f"{with_orth}/{len(genes)} genes have at least one orthologue in these species; "
        f"{sum(len(v) for v in orth.values()):,} orthologues in total")

    out = DOCS / "ortholog_data.js"
    out.write_text(
        HEADER
        + js_const("ORTH_SPECIES", [list(s) for s in SPECIES], indent=0)
        + js_const("ORTH_TREE", TREE)
        + js_const("ORTH", {g: orth[g] for g in genes if orth.get(g)})
        + js_const("ORTH_PARALOGS", {g: paralogs[g] for g in genes if paralogs.get(g)})
        + js_const("ORTH_SEQ", seqs),
        encoding="utf-8")
    log(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) "
        f"in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    build(with_sequences="--no-seq" not in sys.argv[1:],
          refresh="--refresh" in sys.argv[1:])

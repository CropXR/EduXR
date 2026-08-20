"""Generate docs/go_data.js for the W4 GO-enrichment practical page.

The raw sources are far too large to ship to a browser: data/go-basic.obo is
31 MB and data/tair.gaf is 56 MB. This script prunes them down to exactly what
docs/go_enrichment.html needs, which is the biological-process annotations of
the 500 genes that docs/transcriptomics_clustering.html already works with.
That comes to roughly 130 KB.

The gene list and the expression matrix are read straight out of
docs/transcriptomics_clustering.html rather than re-derived from
data/biotic_transcriptomics.txt, so the Week 2 and Week 4 pages can never
disagree about which genes they are showing.

Parsing follows what goatools does, so the numbers on the page match the
notebook:
  - GAF: biological process only (aspect 'P'), NOT-qualified rows dropped
  - obsolete terms dropped, alt_ids resolved to the primary id
  - annotations propagated up is_a and part_of, which is what
    GOEnrichmentStudy(propagate_counts=True) does

Run:

    python tools/build_go_data.py
"""

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"

OBO = DATA / "go-basic.obo"
GAF = DATA / "tair.gaf"
W2_PAGE = DOCS / "transcriptomics_clustering.html"
DST = DOCS / "go_data.js"

BP = "biological_process"

# Evidence codes collapsed to the two buckets the page shows students. The full
# GO evidence-code taxonomy is deliberately not taught here.
EXPERIMENTAL = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP",
}


def read_w2_page():
    """Pull SAMPLES / GENES / EXPR out of the Week 2 clustering page."""
    html = W2_PAGE.read_text(encoding="utf-8")

    def const(name, pattern):
        m = re.search(pattern, html, re.S)
        if not m:
            raise SystemExit(f"could not find `const {name}` in {W2_PAGE}")
        return json.loads(m.group(1))

    samples = const("SAMPLES", r"const SAMPLES = (\[.*?\]);")
    genes = const("GENES", r"const GENES = (\[.*?\]);")
    expr = const("EXPR", r"const EXPR = (\[\[.*?\]\]);")

    if len(expr) != len(genes):
        raise SystemExit(
            f"{W2_PAGE}: EXPR has {len(expr)} rows but GENES has {len(genes)} entries"
        )
    return samples, genes, expr


def read_obo():
    """Return (name, namespace, parents, alt_id, obsolete) maps from go-basic.obo."""
    name, namespace, obsolete = {}, {}, set()
    parents = collections.defaultdict(set)
    alt_id = {}
    cur = None

    with OBO.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"id": None}
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = None            # [Typedef] and friends
                continue
            if cur is None or ": " not in line:
                continue

            key, value = line.split(": ", 1)
            if key == "id":
                cur["id"] = value
            elif cur["id"] is None:
                continue
            elif key == "name":
                name[cur["id"]] = value
            elif key == "namespace":
                namespace[cur["id"]] = value
            elif key == "alt_id":
                alt_id[value] = cur["id"]
            elif key == "is_obsolete" and value == "true":
                obsolete.add(cur["id"])
            elif key == "is_a":
                parents[cur["id"]].add(value.split(" ")[0])
            elif key == "relationship" and value.startswith("part_of "):
                parents[cur["id"]].add(value.split(" ")[1])

    return name, namespace, parents, alt_id, obsolete


def read_gaf(gene_set):
    """Direct BP annotations for the genes we care about: gene -> {term: evidence}."""
    direct = collections.defaultdict(dict)
    symbol = {}

    with GAF.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            gene, sym, qualifier, term, evidence, aspect = (
                parts[1], parts[2], parts[3], parts[4], parts[6], parts[8]
            )
            if gene not in gene_set:
                continue
            symbol.setdefault(gene, sym)
            if aspect != "P" or qualifier.startswith("NOT"):
                continue
            # Keep the most informative evidence if a term is annotated twice.
            if term not in direct[gene] or (
                evidence in EXPERIMENTAL and direct[gene][term] not in EXPERIMENTAL
            ):
                direct[gene][term] = evidence

    return direct, symbol


def ancestors_of(term, parents, cache):
    if term in cache:
        return cache[term]
    cache[term] = set()          # guards against cycles while recursing
    found = set()
    for parent in parents.get(term, ()):
        found.add(parent)
        found |= ancestors_of(parent, parents, cache)
    cache[term] = found
    return found


def depth_of(term, parents, cache):
    """Length of the shortest path to a root, as goatools reports it."""
    if term in cache:
        return cache[term]
    ps = parents.get(term)
    if not ps:
        cache[term] = 0
        return 0
    cache[term] = 0              # cycle guard
    cache[term] = 1 + min(depth_of(p, parents, cache) for p in ps)
    return cache[term]


def build():
    samples, genes, expr = read_w2_page()
    gene_set = set(genes)

    name, namespace, parents, alt_id, obsolete = read_obo()
    print(f"obo: {len(name)} terms, {sum(1 for v in namespace.values() if v == BP)} BP")

    direct_raw, symbol = read_gaf(gene_set)
    print(f"gaf: {len(direct_raw)} of {len(genes)} genes have a BP annotation, "
          f"{sum(len(v) for v in direct_raw.values())} direct annotations")

    # Normalise ids and drop anything obsolete or outside biological process.
    def clean(term):
        term = alt_id.get(term, term)
        if term in obsolete or namespace.get(term) != BP:
            return None
        return term

    direct = collections.defaultdict(dict)
    for gene, terms in direct_raw.items():
        for term, evidence in terms.items():
            t = clean(term)
            if t is not None:
                direct[gene][t] = evidence

    anc_cache = {}
    propagated = collections.defaultdict(set)
    for gene, terms in direct.items():
        for term in terms:
            propagated[gene].add(term)
            propagated[gene] |= {
                a for a in ancestors_of(term, parents, anc_cache)
                if namespace.get(a) == BP and a not in obsolete
            }

    used = sorted(set().union(*propagated.values()))
    index = {term: i for i, term in enumerate(used)}
    depth_cache = {}

    go_terms = [
        [
            term,
            name[term],
            depth_of(term, parents, depth_cache),
            sorted(index[p] for p in parents.get(term, ()) if p in index),
        ]
        for term in used
    ]

    go_direct = {
        gene: sorted(
            ([index[t], ev] for t, ev in terms.items()),
            key=lambda row: go_terms[row[0]][2],
        )
        for gene, terms in sorted(direct.items())
    }
    go_prop = {
        gene: sorted(index[t] for t in terms)
        for gene, terms in sorted(propagated.items())
    }

    pairs = sum(len(v) for v in go_prop.values())
    print(f"propagated: {len(used)} distinct terms, {len(go_prop)} genes, {pairs} pairs")

    dump = lambda obj: json.dumps(obj, separators=(",", ":"))
    body = f"""/* Generated by tools/build_go_data.py -- do not edit by hand.

   Biological-process Gene Ontology data for the {len(genes)} genes of the biotic
   stress transcriptomics dataset (Howard et al. 2013), pruned out of
   data/go-basic.obo and data/tair.gaf.

   GO_TERMS   [go_id, name, depth, [parent indices into GO_TERMS]]
   GO_DIRECT  gene -> [[term index, GAF evidence code], ...]   (as annotated)
   GO_PROP    gene -> [term index, ...]   (annotated terms plus all their ancestors)
   GO_EXPERIMENTAL  evidence codes that mean "measured in the lab"

   SAMPLES / GENES / EXPR are copied from docs/transcriptomics_clustering.html so
   the two pages always show the same dataset. */

const GO_TERMS = {dump(go_terms)};

const GO_DIRECT = {dump(go_direct)};

const GO_PROP = {dump(go_prop)};

const GO_SYMBOL = {dump({g: symbol.get(g, g) for g in genes})};

const GO_EXPERIMENTAL = {dump(sorted(EXPERIMENTAL))};

const SAMPLES = {dump(samples)};

const GENES = {dump(genes)};

const EXPR = {dump(expr)};
"""
    DST.write_text(body, encoding="utf-8")
    print(f"wrote {DST} ({DST.stat().st_size / 1024:.0f} KB)")

    return genes, expr, go_terms, go_prop, index


def reference_enrichment(genes, expr, go_terms, go_prop, index, k=4,
                         metric="correlation", method="average"):
    """Print a trusted enrichment for the page's default settings.

    docs/go_enrichment.html reimplements clustering, Fisher's exact test and the
    Benjamini-Hochberg correction in JavaScript. This printout is what those
    numbers are checked against: open the page at the same settings and the
    tables should agree.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import pdist
    from scipy.stats import hypergeom

    Z = linkage(pdist(np.array(expr), metric), method)
    raw = fcluster(Z, k, "maxclust")

    # The page numbers clusters by the order they first appear in the dendrogram,
    # not by scipy's label order. Renumber so the two can be compared directly.
    remap, labels = {}, np.zeros_like(raw)
    for leaf in leaves_list(Z):
        remap.setdefault(raw[leaf], len(remap) + 1)
    for i, lab in enumerate(raw):
        labels[i] = remap[lab]

    term_genes = collections.defaultdict(set)
    for gene, terms in go_prop.items():
        for t in terms:
            term_genes[t].add(gene)

    population = [g for g in genes if g in go_prop]
    n_pop = len(population)

    print(f"\nreference enrichment  k={k}  metric={metric}  linkage={method}")
    print(f"population: {n_pop} annotated genes of {len(genes)}")

    for cid in sorted(set(labels)):
        study = {g for g, lab in zip(genes, labels) if lab == cid and g in go_prop}
        n_study = len(study)
        size = int((labels == cid).sum())

        raw = []
        for t, carriers in term_genes.items():
            hits = len(carriers & study)
            if hits < 2:
                continue
            p = hypergeom.sf(hits - 1, n_pop, len(carriers), n_study)
            raw.append((p, t, hits, len(carriers)))
        raw.sort()

        # Benjamini-Hochberg, stepping up so the corrected values stay monotone.
        m = len(raw)
        running, corrected = 1.0, []
        for rank in range(m - 1, -1, -1):
            running = min(running, raw[rank][0] * m / (rank + 1))
            corrected.append((running,) + raw[rank][1:])
        corrected.reverse()

        significant = [row for row in corrected if row[0] < 0.05]
        print(f"\n  cluster {cid}: {size} genes ({n_study} annotated), "
              f"{len(significant)} enriched terms")
        for q, t, hits, carriers in significant[:8]:
            go_id, go_name = go_terms[t][0], go_terms[t][1]
            print(f"    {q:.2e}  {go_id}  {go_name[:46]:46s} "
                  f"{hits}/{n_study} vs {carriers}/{n_pop}")


if __name__ == "__main__":
    built = build()
    try:
        reference_enrichment(*built)
    except ImportError as exc:
        print(f"\n(skipping reference enrichment: {exc})")

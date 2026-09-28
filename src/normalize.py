"""Text normalisation for business names and addresses.

Everything is data-independent (no external lookups). Non-Latin scripts are transliterated with
`unidecode`; accents are stripped with Unicode NFKD; legal suffixes are removed from names;
addresses are tokenised, abbreviations expanded and state names collapsed to codes.
"""
from __future__ import annotations
import polars as pl
from unidecode import unidecode

NONLATIN = r"[Ͱ-῿Ⰰ-￿]"
LEGAL = ["inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "pte", "corp",
         "corporation", "co", "company", "plc", "gmbh", "sarl", "sas", "sa", "eurl", "sasu", "sci", "ets",
         "ag", "bv", "nv", "opc", "and", "the"]
TLD = r"(?:com|net|org|in|co\.in|fr|io|biz|info|us|co|shop|store|online)"

ABBREV = {"rd": "road", "st": "street", "dr": "drive", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
          "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "pl": "place",
          "ter": "terrace", "sq": "square", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
          "nr": "near", "opp": "opposite", "mkt": "market", "ctr": "center", "centre": "center",
          "na": "", "nan": "", "null": ""}
ABBREV_COUNTRY = {
    "France": {"r": "rue", "bd": "boulevard", "chem": "chemin", "imp": "impasse", "zi": "zone", "za": "zone",
               "pl": "place", "av": "avenue", "cc": "centre", "ctre": "centre"},
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "lakshadweep": "ld", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "gurugram": "gurgaon"}
STATES = {"US": US_STATES, "India": IN_STATES}


def _translit(df: pl.DataFrame, cols: list) -> pl.DataFrame:
    """Transliterate rows containing non-Latin scripts (a small fraction) via unidecode; others untouched."""
    for c in cols:
        m = pl.col(c).str.contains(NONLATIN).fill_null(False)
        a = df.filter(m)
        if a.height == 0:
            continue
        b = df.filter(~m)
        a = a.with_columns(pl.Series(c, [unidecode(x) for x in a[c].to_list()], dtype=pl.String))
        df = pl.concat([b, a])
    return df


def _ascii(e: pl.Expr) -> pl.Expr:
    """Lowercase, fold accents/ligatures, drop marks."""
    e = e.str.to_lowercase().str.replace_many({"œ": "oe", "æ": "ae", "ø": "o", "ß": "ss", "đ": "d", "ł": "l",
                                               "&": " and ", "’": "", "'": ""})
    return e.str.normalize("NFKD").str.replace_all(r"\p{M}", "")


def _squash(e: pl.Expr) -> pl.Expr:
    """Cheap phonetic squash: merge common spelling/transliteration variants and repeated letters."""
    for a, b in (("ph", "f"), ("ck", "k"), ("sh", "s"), ("th", "t"), ("bh", "b"), ("dh", "d"), ("kh", "k"),
                 ("gh", "g"), ("w", "v"), ("z", "s"), ("q", "k"), ("c", "k"), ("y", "i"), ("x", "ks")):
        e = e.str.replace_all(a, b, literal=True)
    for ch in "abcdefghijklmnopqrstuvwxyz":
        e = e.str.replace_all(f"{ch}{ch}+", ch)
    return e


def _tokens(e: pl.Expr) -> pl.Expr:
    return (e.str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0)))


# legal words as they look after squashing, plus common transliteration spellings
LEGAL_SQ = list(set(pl.DataFrame({"t": LEGAL}).select(_squash(pl.col("t")))["t"].to_list())
                | {"praivet", "praivat", "privet", "limitd", "limitid"})
# extra legal spellings produced by transliteration of Indic scripts (applied to transliterated rows only)
LEGAL_TR = LEGAL_SQ + ["elelpi", "elelp", "elp", "praa", "li", "praibet", "praiveet", "pirai", "piraiveet"]


def normalize(df: pl.DataFrame, country: str) -> pl.DataFrame:
    """df columns: entity_id, business_name, business_address. Returns id + derived columns."""
    df = df.select(pl.col("entity_id").alias("id"), pl.col("business_name").fill_null("").alias("name"),
                   pl.col("business_address").fill_null("").alias("addr"))
    df = df.with_columns(pl.col("name").str.contains(NONLATIN).fill_null(False).alias("_tr"))
    df = _translit(df, ["name", "addr"])

    # ---- names ----
    n = pl.col("name").str.to_lowercase().str.strip_chars()
    dom = rf"^(?:www\.)?([a-z0-9\-]+)\.{TLD}$"
    df = df.with_columns(n.str.contains(dom).alias("is_domain"), n.alias("_n"))
    df = df.with_columns(pl.when(pl.col("is_domain")).then(pl.col("_n").str.replace(dom, "${1}"))
                         .otherwise(pl.col("_n")).alias("_n"))
    e = _ascii(pl.col("_n").str.replace_all(r"\bm/s\b\.?", " ").str.replace_all(r"\.", ""))
    e = e.str.replace_all(r"[^a-z0-9]+", " ")
    df = df.with_columns(_tokens(e).alias("_t"))
    df = df.with_columns(pl.col("_t").list.eval(pl.element().filter(~pl.element().is_in(LEGAL))).alias("_c"))
    df = df.with_columns(pl.when(pl.col("_c").list.len() > 0).then(pl.col("_c")).otherwise(pl.col("_t")).alias("_c"))
    df = df.with_columns(pl.col("_c").list.join(" ").alias("name_norm"))
    df = df.with_columns(_squash(pl.col("name_norm")).alias("name_sq"))
    df = df.with_columns(_tokens(pl.col("name_sq")).alias("_st"))
    df = df.with_columns(pl.when(pl.col("_tr")).then(pl.col("_st").list.eval(pl.element().filter(~pl.element().is_in(LEGAL_TR))))
                         .otherwise(pl.col("_st").list.eval(pl.element().filter(~pl.element().is_in(LEGAL_SQ)))).alias("_sc"))
    df = df.with_columns(pl.when(pl.col("_sc").list.len() > 0).then(pl.col("_sc")).otherwise(pl.col("_st")).alias("name_toks"))
    df = df.with_columns(pl.col("name_toks").list.join(" ").alias("name_sq"))
    df = df.with_columns(pl.col("name_toks").list.join("").alias("name_core"))
    df = df.with_columns(pl.col("name_toks").list.eval(
        pl.element().str.replace_all(r"[aeiou]+", "")).alias("name_skel"))

    # ---- addresses ----
    a = _ascii(pl.col("addr").str.to_lowercase().str.replace_all(r"\bn/a\b", " ").str.replace_all(r"[\.\-/,;:#()\[\]]", " "))
    a = a.str.replace_all(r"[^a-z0-9 ]+", " ")
    df = df.with_columns(a.alias("_a"))
    states = STATES.get(country, {})
    multi = {f" {k} ": f" {v} " for k, v in states.items() if " " in k}
    if multi:
        df = df.with_columns(pl.concat_str(pl.lit(" "), pl.col("_a"), pl.lit(" ")).alias("_a"))
        for _ in range(2):
            df = df.with_columns(pl.col("_a").str.replace_many(multi))
    ab = {**ABBREV, **ABBREV_COUNTRY.get(country, {})}
    single = {k: v for k, v in states.items() if " " not in k}
    df = df.with_columns(_tokens(pl.col("_a")).alias("_at"))
    df = df.with_columns(pl.col("_at").list.eval(
        pl.element().replace(single).replace(ab).str.replace(r"^0+(\d)", "${1}")).alias("_at"))
    df = df.with_columns(pl.col("_at").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).alias("addr_toks"))
    df = df.with_columns(pl.col("addr_toks").list.join(" ").alias("addr_norm"),
                         (pl.col("addr_toks").list.len() == 0).alias("addr_missing"))
    df = df.with_columns(
        pl.col("addr_toks").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.head(3).alias("addr_nums"),
        pl.col("addr_toks").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$"))).alias("addr_alpha"))
    return df.select("id", "name", "addr", "is_domain", "name_norm", "name_sq", "name_toks", "name_core",
                     "name_skel", "addr_norm", "addr_toks", "addr_missing", "addr_nums", "addr_alpha")

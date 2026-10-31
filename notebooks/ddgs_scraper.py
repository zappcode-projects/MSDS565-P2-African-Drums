"""
DuckDuckGo image scraper for MSDS 565 P2.

Build a labeled image dataset by searching DuckDuckGo Images, downloading the
results, filtering out images that are too small or the wrong shape, and saving
what survives into one folder per class.

    from pathlib import Path
    from ddgs_scraper import build_image_dataset

    records, summary = build_image_dataset(
        class_queries={
            "hammer": ["claw hammer", "ball peen hammer", "sledgehammer"],
            "wrench": ["adjustable wrench", "socket wrench", "pipe wrench"],
        },
        save_dir=Path.home() / "MSDS565_data" / "P2_images",   # outside your repo
        images_per_class=250,
    )

`records` is one row per saved image and is what you turn into the manifest CSV.

Requires the `ddgs` package:  pip install ddgs

A note on where the results come from: DuckDuckGo's image results are Bing's
image index, and the `ddgs` package can ask for them either through
DuckDuckGo's front end ("duckduckgo") or from Bing directly ("bing"). This
module asks Bing directly by default — see `search_images` for why.
"""

import random
import re
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from ddgs import DDGS


# ================================================================
# 1. Validating class names and queries
# ================================================================
# A class label is used in two places on disk:
#   1. As a sub-directory name:  save_dir/<label>/
#   2. As a filename prefix:     save_dir/<label>/<label>_000.jpg
# The same character rules apply to both, so one validator covers both.

# Characters that are illegal in folder/file names on Windows or Unix
_ILLEGAL_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

# Reserved device names on Windows (case-insensitive)
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Conservative cross-platform limit, well under 255 so there is room for the
# "_000.jpg" suffix (8 characters) that gets appended to filenames.
_MAX_NAME_LEN = 200


def validate_label_as_folder_name(label: str) -> str:
    """
    Check that a class label can be used as both a folder name and a filename
    prefix. Returns the stripped label on success; raises ValueError otherwise.
    """
    if not isinstance(label, str):
        raise TypeError(f"Label must be a str, got {type(label).__name__!r}: {label!r}")

    stripped = label.strip()

    if not stripped:
        raise ValueError(f"Label is empty or whitespace-only: {label!r}")

    illegal = re.findall(_ILLEGAL_CHARS, stripped)
    if illegal:
        chars = ", ".join(repr(c) for c in sorted(set(illegal)))
        raise ValueError(
            f"Label {stripped!r} contains characters that are illegal in folder "
            f"and file names: {chars}"
        )

    if stripped.upper() in _WINDOWS_RESERVED:
        raise ValueError(
            f"Label {stripped!r} is a reserved Windows device name and cannot be "
            "used as a folder name or filename prefix."
        )

    if len(stripped) > _MAX_NAME_LEN:
        raise ValueError(
            f"Label {stripped!r} is {len(stripped)} characters long; folder and "
            f"file names must be at most {_MAX_NAME_LEN} characters."
        )

    return stripped


def validate_class_queries(class_queries: dict) -> dict:
    """
    Validate the whole {label: [queries]} mapping before any network calls.

    Failing early matters: a scrape can run for many minutes, and discovering a
    bad folder name after the fact means doing it all again.
    """
    if not isinstance(class_queries, dict):
        raise TypeError(
            "class_queries must be a dict mapping a class label to a list of "
            f"search queries, got {type(class_queries).__name__!r}"
        )
    if not class_queries:
        raise ValueError("class_queries is empty.")

    cleaned = {}
    for label, queries in class_queries.items():
        clean_label = validate_label_as_folder_name(label)

        if isinstance(queries, str):
            raise TypeError(
                f"Queries for class {clean_label!r} must be a list of strings, not a "
                f"single string. Did you mean [{queries!r}]?"
            )
        if not queries:
            raise ValueError(f"Class {clean_label!r} has no search queries.")

        clean_queries = []
        for i, q in enumerate(queries):
            if not isinstance(q, str) or not q.strip():
                raise ValueError(
                    f"Query {i} for class {clean_label!r} is empty or not a string: {q!r}"
                )
            clean_queries.append(q.strip())

        if len(set(clean_queries)) != len(clean_queries):
            raise ValueError(f"Class {clean_label!r} has duplicate queries.")

        cleaned[clean_label] = clean_queries

    return cleaned


# ================================================================
# 2. Image quality filters
# ================================================================

def validate_aspect_ratio_range(aspect_ratio_range):
    if not isinstance(aspect_ratio_range, (tuple, list)) or len(aspect_ratio_range) != 2:
        raise ValueError(
            "aspect_ratio_range must be a tuple/list of length 2, e.g. (0.5, 2.0)"
        )
    min_ratio, max_ratio = aspect_ratio_range
    if min_ratio <= 0 or max_ratio <= 0:
        raise ValueError("aspect_ratio_range values must be positive")
    if min_ratio > max_ratio:
        raise ValueError("aspect_ratio_range must satisfy min <= max")
    return float(min_ratio), float(max_ratio)


def validate_resolution_threshold(min_resolution):
    if not isinstance(min_resolution, (tuple, list)) or len(min_resolution) != 2:
        raise ValueError(
            "min_resolution must be a tuple/list of length 2, e.g. (200, 200)"
        )
    min_w, min_h = min_resolution
    if min_w <= 0 or min_h <= 0:
        raise ValueError("min_resolution values must be positive")
    return int(min_w), int(min_h)


def passes_image_filters(
    img: Image.Image,
    aspect_ratio_range=(0.5, 2.0),
    min_resolution=(200, 200),
) -> tuple[bool, list[str]]:
    """
    Return (passed, failed_reasons) for a single image. `failed_reasons` is an
    empty list when the image passes everything.

    These filters are measured on the image actually downloaded, not on the
    dimensions the search engine reported — those are frequently wrong.
    """
    min_ratio, max_ratio = validate_aspect_ratio_range(aspect_ratio_range)
    min_w, min_h = validate_resolution_threshold(min_resolution)

    w, h = img.size
    ratio = w / max(h, 1)
    reasons = []

    if w < min_w or h < min_h:
        reasons.append("resolution")
    if not (min_ratio <= ratio <= max_ratio):
        reasons.append("aspect_ratio")

    return len(reasons) == 0, reasons


# ================================================================
# 3. Downloading one image
# ================================================================
# Image URLs point at arbitrary websites, not at the search engine. Many will
# be slow, dead, blocking scrapers, or serving something that is not an image.
#
# Each URL gets exactly ONE attempt, with a hard time limit. It is tempting to
# retry failures with backoff, but here that is the wrong call: a dead host is
# still dead ten seconds later, a 403 is still a 403, and every search returns
# far more candidates than you need, so skipping one costs nothing. Retrying
# every failure 3 times with backoff turns each dead URL into ~40 seconds of
# waiting, and a page of them makes the scrape look frozen.
#
# Retry-with-backoff IS the right tool for the search itself (section 4): that
# is one service you depend on, and when it throttles you it does recover.

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

# Seconds to wait for a server to accept the connection at all. Kept short: a
# host that has not answered in 5 seconds is almost never going to.
CONNECT_TIMEOUT = 5

# Largest download accepted, in bytes. Real photos are well under this; files
# bigger than it are print-resolution originals that take ages to fetch.
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def get_default_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        # Ask for formats PIL can open, so servers that pick a format by the
        # Accept header do not hand back one it cannot read.
        "Accept": "image/jpeg,image/png,image/webp,image/*;q=0.8",
    }


def download_image(url: str, headers=None, timeout=10) -> Image.Image | None:
    """
    Download a single image URL and return it as an RGB PIL image.

    Returns None — never raises — when anything goes wrong: the host is dead
    or refuses us, the URL is not an image, the download takes longer than
    `timeout` seconds in total, or the bytes will not open. All of these are
    normal and common when scraping; the caller just moves on to the next URL.
    """
    if headers is None:
        headers = get_default_headers()

    start = time.monotonic()
    data = bytearray()

    try:
        # requests' own timeout only limits how long to wait for the NEXT
        # bytes — a server that keeps trickling out a few bytes a second never
        # trips it, and can hold one download for minutes. So the body is read
        # in pieces (stream=True) and the clock is checked after every piece.
        with requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, timeout), stream=True) as resp:
            if resp.status_code != 200:
                return None     # 403, 404, 429, 5xx ... skip it, do not retry

            if not resp.headers.get("Content-Type", "").startswith("image/"):
                return None     # an HTML page, a redirect to a login, etc.

            while True:
                # read1 returns whatever bytes have arrived (up to 64 KB)
                # instead of waiting for a full 64 KB, so a slow server cannot
                # stall us between clock checks. Older urllib3 (before 2.3)
                # has no read1; there, read a whole chunk instead.
                if hasattr(resp.raw, "read1"):
                    chunk = resp.raw.read1(64 * 1024, decode_content=True)
                else:
                    chunk = resp.raw.read(64 * 1024, decode_content=True)

                if not chunk:
                    break       # download finished
                data.extend(chunk)

                if len(data) > MAX_IMAGE_BYTES:
                    return None
                if time.monotonic() - start > timeout:
                    return None

    except Exception:
        # Dead host, refused connection, timeout, bad SSL certificate, a URL
        # requests cannot parse ... Caught broadly on purpose: one bad URL must
        # never stop the scrape. (Ctrl+C / interrupting the kernel still works —
        # KeyboardInterrupt is not an Exception.)
        return None

    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        # Truncated downloads, formats PIL cannot read, HTML error pages served
        # with an image content-type — all land here.
        return None


# ================================================================
# 4. Searching for images
# ================================================================

# ---------------------------------------------------------------
# 4.1 Checking that results are about the query
# ---------------------------------------------------------------
# Every so often the search engine answers with a page of results for some
# OTHER search entirely — ask for "acoustic guitar" and get thirty-five stills
# from a 1991 movie. Nothing about the response says anything went wrong; the
# images download fine and pass every size filter, so without a check they
# land in your class folder looking like real data.
#
# The check is deliberately crude: does the result's title, page URL, or image
# URL contain at least ONE word from the query? A genuine result almost always
# does (on real queries, 34–35 out of 35 pass). A wrong-query page almost never
# does. This is a guard against garbage, NOT a check that the picture shows the
# right thing — an electric guitar titled "guitar" passes a query for
# "acoustic guitar". Judging that is your job in Notebook 2.

# Words too generic to say anything about whether a result is on topic.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "stock",
    "photo", "photos", "image", "images", "picture", "pictures",
}


def query_keywords(query: str) -> set[str]:
    """
    The words of a query worth checking results against: lower-cased, at
    least 3 letters, not a stopword, and with a plural ending trimmed so that
    "wrenches" still matches a title that says "wrench".
    """
    keywords = set()
    for word in re.split(r"[^a-z0-9]+", query.lower()):
        if len(word) < 3 or word in _STOPWORDS:
            continue
        if word.endswith("es") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        keywords.add(word)
    return keywords


def result_mentions_query(result: dict, keywords: set[str]) -> bool:
    """
    True if any keyword appears in the result's title, page URL, or image URL.

    Spaces and punctuation are squeezed out first, so "sledgehammer" matches
    "Sledge Hammer" and "sledge-hammer.jpg".
    """
    if not keywords:
        return True     # nothing to check against (e.g. a one- or two-letter query)

    text = " ".join([
        result.get("title") or "",
        result.get("url") or "",
        result.get("image") or "",
    ])
    squeezed = re.sub(r"[^a-z0-9]", "", text.lower())
    return any(word in squeezed for word in keywords)


# ---------------------------------------------------------------
# 4.2 One page of results, with retries
# ---------------------------------------------------------------
# A search engine that is throttling you does not say so. Both engines report
# "No results found" instead — which looks exactly like a query that genuinely
# matched nothing. Because a real query almost never returns nothing, an empty
# page is retried with backoff before it is believed. So is a page that fails
# the check in 4.1. Without the retry a whole class can silently end up empty,
# or full of someone else's search.

def search_one_page(
    query: str,
    page: int = 1,
    backend: str = "bing",
    safesearch: str = "moderate",
    region: str = "us-en",
    require_query_match: bool = True,
    retries: int = 4,
    backoff_base: float = 3.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Fetch one page of image results for `query`, retrying a failed, empty, or
    off-topic page with exponential backoff. Returns [] if every attempt fails.

    When `require_query_match` is True, results that do not mention the query
    are dropped, and a page where fewer than half of them mention it is treated
    as a bad response and retried.
    """
    keywords = query_keywords(query)

    for attempt in range(retries):
        off_topic = False   # set below if this attempt fails the 4.1 check
        try:
            # A fresh DDGS() each attempt, so a retry is a clean new session.
            results = DDGS().images(
                query=query,
                page=page,
                max_results=None,       # None = everything on this page
                backend=backend,
                safesearch=safesearch,
                region=region,
            )
            problem = "no results"
        except Exception as e:
            results = []
            problem = f"{type(e).__name__}: {e}"

        if results and not require_query_match:
            return results

        if results:
            on_topic = [r for r in results if result_mentions_query(r, keywords)]

            if len(on_topic) >= len(results) / 2:
                n_dropped = len(results) - len(on_topic)
                if verbose and n_dropped > 0:
                    print(f"    page {page}: dropped {n_dropped} of {len(results)} "
                          "results that do not mention the query")
                return on_topic

            off_topic = True
            problem = (f"only {len(on_topic)} of {len(results)} results mention "
                       f"{query!r} — the engine answered a different search")

        if attempt < retries - 1:
            sleep = (backoff_base ** attempt) + random.uniform(1.0, 3.0)
            if verbose:
                print(f"    page {page}: {problem} — retrying in {sleep:.1f}s "
                      f"({attempt + 1}/{retries - 1})")
            time.sleep(sleep)

    if verbose:
        print(f"    page {page}: giving up after {retries} attempts ({problem})")
        if off_topic:
            print("    If this query's words genuinely do not appear in image titles "
                  "(another language, a brand nickname, ...),\n"
                  "    pass require_query_match=False — then check this class by eye.")
    return []


# ---------------------------------------------------------------
# 4.3 A whole search: as many pages as it takes
# ---------------------------------------------------------------
# One request returns one page — 35 results from Bing, ~100 from DuckDuckGo —
# no matter how many you ask for. Getting 200 results means asking for page 1,
# then page 2, and so on, and stopping once there are enough.

# Safety stop for paging, so a search can never page forever.
MAX_PAGES = 20


def search_images(
    query: str,
    max_results: int = 100,
    backend: str = "bing",
    safesearch: str = "moderate",
    region: str = "us-en",
    require_query_match: bool = True,
    retries: int = 4,
    backoff_base: float = 3.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run one image search and return up to `max_results` result records.

    Each record is a dict with keys: image (full-size URL), thumbnail, url
    (the page the image was found on), title, width, height, source.

    Parameters
    ----------
    query : str
        The search term.
    max_results : int, default 100
        How many results to collect. Fetched a page at a time until there are
        this many, or the engine runs out. You can get fewer.
    backend : {"bing", "duckduckgo", "auto"}, default "bing"
        Which engine the `ddgs` package asks. DuckDuckGo's image results ARE
        Bing's, so the two return the same kind of thing. "bing" asks Bing
        directly and is the reliable one. "duckduckgo" goes through
        DuckDuckGo's front end, and with ddgs 9.16 / primp 2.0 it fails on every
        request ("malformed headers"). "auto" picks an engine at random on every
        request, which mixes engines from one page to the next and makes paging
        skip and repeat results. Pin one engine so your run is consistent.
    safesearch : {"on", "moderate", "off"}, default "moderate"
        Content filter. Leave this at "moderate" unless you have a reason not
        to. NOTE: only the "duckduckgo" backend passes this on. The "bing"
        backend ignores it and gets Bing's own default filtering, which is
        Moderate.
    region : str, default "us-en"
        Region/language code. Changing it changes what you get back — worth
        noting in your writeup. Like safesearch, the "bing" backend ignores it.
    require_query_match : bool, default True
        Drop results that do not mention the query, and retry pages that are
        mostly off-topic (see section 4.1). Turn off only if your query's words
        genuinely do not appear in image titles — and then check the class by eye.
    retries : int, default 4
        How many times to try each page before giving up on it.
    backoff_base : float, default 3.0
        Base of the exponential backoff between retries, in seconds.
    """
    collected = []
    seen_urls = set()

    for page in range(1, MAX_PAGES + 1):
        page_results = search_one_page(
            query,
            page=page,
            backend=backend,
            safesearch=safesearch,
            region=region,
            require_query_match=require_query_match,
            retries=retries,
            backoff_base=backoff_base,
            verbose=verbose,
        )

        # Pages overlap a little; keep only images we have not seen yet.
        n_new = 0
        for result in page_results:
            image_url = result.get("image")
            if image_url and image_url not in seen_urls:
                seen_urls.add(image_url)
                collected.append(result)
                n_new += 1

        if len(collected) >= max_results:
            break
        if n_new == 0:
            break       # the engine has run out (or kept failing) — stop paging

        time.sleep(random.uniform(1.0, 2.0))    # be polite between pages

    return collected[:max_results]


# ================================================================
# 5. Scraping one class
# ================================================================

def scrape_class(
    label: str,
    queries: list[str],
    save_dir: Path,
    images_per_class: int = 250,
    results_per_query: int = 200,
    aspect_ratio_range: tuple = (0.5, 2.0),
    min_resolution: tuple = (200, 200),
    per_image_timeout: int = 10,
    backend: str = "bing",
    safesearch: str = "moderate",
    region: str = "us-en",
    require_query_match: bool = True,
    query_delay: tuple = (2.0, 5.0),
    verbose: bool = True,
) -> list[dict]:
    """
    Scrape every query for one class into save_dir/<label>/ and return one
    record per saved image.

    Every query is searched, and the downloads then take turns between them:
    the first result of each query, then the second result of each, and so on,
    until the class has images_per_class images. Each query therefore supplies
    a roughly equal share of the class. If one query runs dry or most of its
    links are dead, the others make up the difference.
    """
    class_dir = save_dir / label
    class_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: search every query
    # ---------------------------------------------------------------
    # Downloading query 1's results until the class is full would fill most
    # classes from query 1 alone, and the other queries would never be used.
    # So all the searching happens first, before any downloading.
    results_by_query = []

    for qi, query in enumerate(queries):
        if verbose:
            print(f"  query {qi + 1}/{len(queries)}: {query!r}")

        results = search_images(
            query,
            max_results=results_per_query,
            backend=backend,
            safesearch=safesearch,
            region=region,
            require_query_match=require_query_match,
            verbose=verbose,
        )
        results_by_query.append(results)

        if verbose:
            print(f"    {len(results)} results returned")

        # Pause between searches so the search engine does not start refusing us.
        if qi < len(queries) - 1:
            time.sleep(random.uniform(*query_delay))

    # ---------------------------------------------------------------
    # Step 2: interleave the queries' results
    # ---------------------------------------------------------------
    # Result 0 of every query, then result 1 of every query, then result 2 ...
    # Each entry keeps the query it came from, for the manifest.
    candidates = []
    longest = max(len(results) for results in results_by_query)

    for i in range(longest):
        for query, results in zip(queries, results_by_query):
            if i < len(results):
                candidates.append((query, results[i]))

    # ---------------------------------------------------------------
    # Step 3: download in that order until the class is full
    # ---------------------------------------------------------------
    records = []
    seen_urls = set()   # dedupe across queries — overlapping queries return overlapping images
    headers = get_default_headers()

    # What happened to each candidate, so the gap between "returned" and
    # "saved" can be explained rather than guessed at.
    n_tried = 0
    n_failed_download = 0
    n_failed_filters = 0

    if verbose:
        print(f"  downloading from {len(candidates)} results, taking turns between queries")

    for query, result in candidates:
        if len(records) >= images_per_class:
            break

        image_url = result.get("image")
        if not image_url or image_url in seen_urls:
            continue
        seen_urls.add(image_url)

        # A heartbeat, so a run of slow or dead hosts does not look frozen.
        if verbose and n_tried > 0 and n_tried % 10 == 0:
            print(f"    ...tried {n_tried}, class has {len(records)}/{images_per_class}")
        n_tried += 1

        img = download_image(image_url, headers=headers, timeout=per_image_timeout)
        if img is None:
            n_failed_download += 1
            continue

        passed, _reasons = passes_image_filters(
            img,
            aspect_ratio_range=aspect_ratio_range,
            min_resolution=min_resolution,
        )
        if not passed:
            n_failed_filters += 1
            continue

        # Save the image itself. Do NOT round-trip through matplotlib —
        # that re-renders the picture into a figure and changes both the
        # resolution and the framing of what you are training on.
        filename = f"{label}_{len(records):04d}.jpg"
        img.save(class_dir / filename, format="JPEG", quality=92)

        records.append({
            "filename": filename,
            "label": label,
            "query": query,
            "image_url": image_url,
            "source_url": result.get("url", ""),
            "title": result.get("title", ""),
            "width": img.size[0],
            "height": img.size[1],
        })

    if verbose:
        print(f"    tried {n_tried}: {n_failed_download} failed to download, "
              f"{n_failed_filters} failed the size/shape filters, {len(records)} saved")

        # How many images each query ended up contributing.
        for query in queries:
            n_from_query = sum(1 for r in records if r["query"] == query)
            note = "  <- none; likely rate limited, re-run this class later" if n_from_query == 0 else ""
            print(f"    {n_from_query:>5} from {query!r}{note}")

        print(f"  -> {len(records)} images saved to {class_dir}")

    return records


# ================================================================
# 6. Building the whole dataset
# ================================================================

def build_image_dataset(
    class_queries: dict,
    save_dir: str | Path,
    images_per_class: int = 250,
    results_per_query: int = 200,
    aspect_ratio_range: tuple = (0.5, 2.0),
    min_resolution: tuple = (200, 200),
    per_image_timeout: int = 10,
    backend: str = "bing",
    safesearch: str = "moderate",
    region: str = "us-en",
    require_query_match: bool = True,
    query_delay: tuple = (2.0, 5.0),
    continue_on_error: bool = True,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Download a labeled image dataset from an image search engine.

    One folder per class is created inside save_dir. Every class may have
    several search queries, and results from all of a class's queries land in
    the same folder — using several queries per class is how you get a class
    that looks like the whole category instead of one narrow slice of it.
    Every query is used: downloads take turns between a class's queries, so
    each supplies a roughly equal share of its images (see `scrape_class`).

    Parameters
    ----------
    class_queries : dict[str, list[str]]
        Maps each class label to its list of search queries. The label becomes
        the folder name and the filename prefix, so it must be a legal filename.
    save_dir : str or Path
        Root output directory. Created if it does not exist. Put it in your
        local class data folder, OUTSIDE your repo — scraped images are never
        committed.
    images_per_class : int, default 250
        Stop saving once a class reaches this many images. Scrape well beyond
        the number you need to end up with — you will lose a meaningful fraction
        to junk and duplicates during cleaning.
    results_per_query : int, default 200
        How many search results to collect per query, fetched a page at a time.
        Not how many you keep: many will fail to download or fail the filters.
    aspect_ratio_range : tuple of (float, float), default (0.5, 2.0)
        Inclusive (min, max) bounds on width/height. Rejects extreme panoramas
        and very tall images, which are usually banners or infographics.
    min_resolution : tuple of (int, int), default (200, 200)
        Minimum acceptable (width, height) in pixels, measured on the downloaded
        image. Anything smaller is a thumbnail and will look terrible resized up.
    per_image_timeout : int, default 10
        Seconds allowed for a single image download, start to finish, before
        giving up on it. Each image gets one attempt — see section 3.
    backend : {"bing", "duckduckgo", "auto"}, default "bing"
        Which engine to search. See `search_images` for why "bing" is the default.
    safesearch : {"on", "moderate", "off"}, default "moderate"
        Content filter. Ignored by the "bing" backend, which always applies
        Bing's default (Moderate).
    region : str, default "us-en"
        Region/language code for the search. Also ignored by "bing".
    require_query_match : bool, default True
        Drop search results that do not mention the query, and retry result
        pages that are mostly off-topic. See section 4.1.
    query_delay : tuple of (float, float), default (2.0, 5.0)
        Random sleep range in seconds between queries, to avoid rate limiting.
    continue_on_error : bool, default True
        If True, a class that fails is logged and the run continues. If False,
        the first failure raises.
    verbose : bool, default True
        Print progress as the scrape runs.

    Returns
    -------
    records : list of dict
        One entry per saved image, with keys: filename, label, query,
        image_url, source_url, title, width, height. This is what you build
        your manifest CSV from.
    summary : dict
        Per-class counts and status, plus "total_saved" and "save_dir".
    """
    # Validate everything before touching the network.
    cleaned = validate_class_queries(class_queries)
    validate_aspect_ratio_range(aspect_ratio_range)
    validate_resolution_threshold(min_resolution)

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    summary = {"classes": {}, "total_saved": 0, "save_dir": str(save_dir)}

    for i, (label, queries) in enumerate(cleaned.items()):
        if verbose:
            print(f"\n[{i + 1}/{len(cleaned)}] class {label!r} — {len(queries)} queries")

        status, error_msg, records = "ok", None, []

        try:
            records = scrape_class(
                label=label,
                queries=queries,
                save_dir=save_dir,
                images_per_class=images_per_class,
                results_per_query=results_per_query,
                aspect_ratio_range=aspect_ratio_range,
                min_resolution=min_resolution,
                per_image_timeout=per_image_timeout,
                backend=backend,
                safesearch=safesearch,
                region=region,
                require_query_match=require_query_match,
                query_delay=query_delay,
                verbose=verbose,
            )
        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  FAILED: class {label!r} | {error_msg}")
            if not continue_on_error:
                raise

        # A class that finishes with nothing is almost always rate limiting, not
        # a genuinely unsearchable category. Flag it rather than reporting "ok",
        # so an empty class cannot slide past unnoticed.
        if status == "ok" and len(records) == 0:
            status = "empty"
            error_msg = "no images saved — likely rate limited; wait and re-run this class"

        all_records.extend(records)
        summary["classes"][label] = {
            "saved": len(records),
            "queries": len(queries),
            "status": status,
            "error": error_msg,
        }
        summary["total_saved"] += len(records)

    if verbose:
        print(f"\nDone. {summary['total_saved']} images saved to {save_dir}")
        for label, info in summary["classes"].items():
            flag = "" if info["status"] == "ok" else f"  [{info['status']}]"
            print(f"  {label:<24} {info['saved']:>5}{flag}")

    return all_records, summary

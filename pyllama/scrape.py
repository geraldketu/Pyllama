"""Scrape the official Python documentation (docs.python.org) into structured JSON.

Each crawled page is parsed into:

* ``sections`` - heading, prose paragraphs and code examples for every section
* ``api_entries`` - documented functions, classes, methods and attributes
  (Sphinx ``dl.py`` blocks) with their signature, description and examples

Usage::

    python -m pyllama.scrape --out data/raw/python_docs.jsonl --max-pages 400
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://docs.python.org/3/"
DEFAULT_SEEDS = ("library/index.html", "tutorial/index.html", "howto/index.html", "reference/index.html")
ALLOWED_PREFIXES = ("library/", "tutorial/", "howto/", "reference/")
USER_AGENT = "PyLLaMa-docs-scraper/0.1 (research dataset builder)"
API_KINDS = ("function", "class", "method", "classmethod", "staticmethod", "attribute", "exception", "data")


@dataclass
class Section:
    title: str
    anchor: str
    paragraphs: list[str] = field(default_factory=list)
    code: list[str] = field(default_factory=list)


@dataclass
class ApiEntry:
    name: str
    kind: str
    signature: str
    description: str
    code: list[str] = field(default_factory=list)


@dataclass
class Page:
    url: str
    title: str
    sections: list[Section]
    api_entries: list[ApiEntry]


def _clean(text: str) -> str:
    text = text.replace("¶", "")  # Sphinx permalink pilcrow
    return re.sub(r"\s+", " ", text).strip()


def _is_highlight(node: Tag) -> bool:
    return node.name == "div" and any(c.startswith("highlight-") for c in node.get("class", []))


def _code_blocks(node: Tag) -> list[str]:
    """Code examples in ``node``: the node itself if it is a highlight block, else its descendants."""
    pres = [node.find("pre")] if _is_highlight(node) else node.select("div[class*='highlight-'] pre")
    return [pre.get_text().rstrip() for pre in pres if pre is not None and pre.get_text().strip()]


def _parse_api_entry(dl: Tag) -> ApiEntry | None:
    kind = next((k for k in API_KINDS if k in dl.get("class", [])), None)
    dt = dl.find("dt", recursive=False)
    dd = dl.find("dd", recursive=False)
    if kind is None or dt is None or dd is None:
        return None
    name = dt.get("id") or _clean(dt.get_text())
    for link in dt.select("a.headerlink"):
        link.decompose()
    signature = _clean(dt.get_text())

    # Only the entry's own prose: skip nested API entries (e.g. methods of a class).
    paragraphs = []
    code = []
    for child in dd.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "dl" and "py" in child.get("class", []):
            continue
        if child.name == "p":
            paragraphs.append(_clean(child.get_text()))
        code.extend(_code_blocks(child))
    description = "\n\n".join(p for p in paragraphs if p)
    if not description:
        return None
    return ApiEntry(name=name, kind=kind, signature=signature, description=description, code=code)


def parse_page(html: str, url: str) -> Page:
    """Parse one Sphinx documentation page."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.body") or soup.body or soup
    h1 = body.find("h1")
    title = _clean(h1.get_text()) if h1 else _clean(soup.title.get_text() if soup.title else url)

    sections: list[Section] = []
    for sec in body.find_all("section"):
        heading = sec.find(re.compile(r"^h[1-6]$"))
        if heading is None:
            continue
        current = Section(title=_clean(heading.get_text()), anchor=sec.get("id", ""))
        # Direct content only; nested sections are visited on their own.
        for child in sec.children:
            if not isinstance(child, Tag) or child.name == "section":
                continue
            if child.name == "dl" and "py" in child.get("class", []):
                continue
            if child.name == "p":
                text = _clean(child.get_text())
                if text:
                    current.paragraphs.append(text)
            current.code.extend(_code_blocks(child))
        if current.paragraphs or current.code:
            sections.append(current)

    api_entries = []
    for dl in body.select("dl.py"):
        entry = _parse_api_entry(dl)
        if entry is not None:
            api_entries.append(entry)

    return Page(url=url, title=title, sections=sections, api_entries=api_entries)


def extract_links(html: str, url: str) -> list[str]:
    """Return in-scope documentation links found on a page (fragments removed)."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("div.body a[href], div.toctree-wrapper a[href]"):
        target, _ = urldefrag(urljoin(url, a["href"]))
        if target == url or not target.startswith(BASE_URL) or not target.endswith(".html"):
            continue
        rel = target[len(BASE_URL):]
        if rel.startswith(ALLOWED_PREFIXES):
            links.append(target)
    return links


class Crawler:
    def __init__(self, delay: float = 0.5, cache_dir: Path | None = None):
        self.delay = delay
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        retry = Retry(total=5, connect=5, read=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.robots = RobotFileParser(urljoin(BASE_URL, "/robots.txt"))
        try:
            self.robots.read()
        except Exception:  # network issues: fall back to allowing, we still rate-limit
            self.robots = None

    def fetch(self, url: str) -> str | None:
        if self.cache_dir is not None:
            cached = self.cache_dir / urlparse(url).path.strip("/").replace("/", "__")
            if cached.exists():
                return cached.read_text(encoding="utf-8")
        if self.robots is not None and not self.robots.can_fetch(USER_AGENT, url):
            return None
        try:
            resp = self.session.get(url, timeout=30)
        except requests.RequestException as exc:
            print(f"  ! skipping {url}: {exc}")
            return None
        finally:
            time.sleep(self.delay)
        if resp.status_code != 200:
            return None
        resp.encoding = "utf-8"
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(resp.text, encoding="utf-8")
        return resp.text

    def crawl(self, seeds: list[str], max_pages: int):
        queue = deque(urljoin(BASE_URL, s) for s in seeds)
        seen = set(queue)
        count = 0
        while queue and count < max_pages:
            url = queue.popleft()
            html = self.fetch(url)
            if html is None:
                continue
            count += 1
            yield parse_page(html, url)
            for link in extract_links(html, url):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw/python_docs.jsonl"))
    ap.add_argument("--max-pages", type=int, default=400)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--seeds", nargs="*", default=list(DEFAULT_SEEDS))
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    crawler = Crawler(delay=args.delay, cache_dir=args.cache_dir)
    n_pages = n_sections = n_api = 0
    with args.out.open("w", encoding="utf-8") as f:
        for page in crawler.crawl(args.seeds, args.max_pages):
            f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
            n_pages += 1
            n_sections += len(page.sections)
            n_api += len(page.api_entries)
            print(f"[{n_pages}] {page.title} ({len(page.sections)} sections, {len(page.api_entries)} API entries)")
    print(f"Done: {n_pages} pages, {n_sections} sections, {n_api} API entries -> {args.out}")


if __name__ == "__main__":
    main()

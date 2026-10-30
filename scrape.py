"""scrape.py - download GOV.UK pages with requests + BeautifulSoup and turn them into clean
sentence lists. Run once:  python scrape.py
GOV.UK content is published under the Open Government Licence v3.0."""
import json
import re
import time

import requests
from bs4 import BeautifulSoup

import config
from metrics import extract_anchors, split_sentences


def extract_doc(html: str, url: str):
    """Parse one page. Returns a dict, or None if the page is unsuitable for the experiment."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("main#content") or soup.find("main") or soup.body   # GOV.UK main content region
    if main is None:
        return None
    # remove page furniture that is not guidance content
    for tag in main.select("script, style, nav, footer, .gem-c-contents-list, "
                           ".gem-c-feedback, .gem-c-print-link, [role=complementary]"):
        tag.decompose()                       # decompose() deletes the element from the tree
    h1 = main.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else url

    lines = []
    for el in main.select("p, li, tr"):       # paragraphs, list items, table rows carry the facts
        if el.name == "p" and el.find_parent("li"):
            continue                          # paragraph inside a list item: the <li> already has it
        if el.name == "tr":                   # flatten a table row to "cell - cell - cell"
            text = " - ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"]))
        else:
            text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()   # collapse whitespace / nbsp
        if text and text[-1] not in ".!?:":
            text += "."                       # make list items and rows proper sentences
        lines.append(text)

    sentences, seen, words = [], set(), 0
    for s in split_sentences("\n".join(lines)):
        if len(s.split()) < 4 or s in seen:   # drop fragments and exact duplicates
            continue
        if words + len(s.split()) > config.MAX_DOC_WORDS:
            break                             # truncate at a sentence boundary
        seen.add(s)
        sentences.append(s)
        words += len(s.split())

    if len(sentences) < config.MIN_SENTENCES:                      # too short to shard across agents
        return None
    if len(extract_anchors(" ".join(sentences))) < config.MIN_ANCHORS:   # too few measurable facts
        return None
    return {"url": url, "title": title, "sentences": sentences}


def main():
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT   # identify ourselves
    docs = []
    for url in config.SEED_URLS:
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()              # raise on 404/500 so we can skip bad URLs
        except requests.RequestException as e:
            print(f"SKIP {url}: {e}")
            continue
        doc = extract_doc(r.text, url)
        if doc is None:
            print(f"SKIP {url}: too short / too few numbers")
        else:
            doc["id"] = f"doc{len(docs):02d}"
            docs.append(doc)
            print(f"OK   {url}: {len(doc['sentences'])} sentences")
        time.sleep(config.REQUEST_DELAY_S)    # be polite between requests
    config.DOCS_PATH.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(docs)} documents to {config.DOCS_PATH}")


if __name__ == "__main__":
    main()
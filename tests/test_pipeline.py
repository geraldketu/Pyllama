import json

import pytest

from pyllama.benchmark import cosine, extract_code, load_problems
from pyllama.build_dataset import build, split
from pyllama.prompts import build_example, build_prompt
from pyllama.scrape import extract_links, parse_page

URL = "https://docs.python.org/3/library/example.html"
HTML = """
<html><head><title>example</title></head><body><div class="body">
<section id="example-module">
  <h1>example — An example module<a class="headerlink" href="#example-module">¶</a></h1>
  <p>This module provides   example helpers for testing the parser.</p>
  <div class="highlight-python3 notranslate"><div class="highlight"><pre>import example
example.greet("world")</pre></div></div>
  <dl class="py function">
    <dt class="sig sig-object py" id="example.greet">example.greet(<em>name</em>)<a class="headerlink" href="#example.greet">¶</a></dt>
    <dd><p>Return a greeting for <em>name</em> as a string, suitable for printing to the console.</p>
      <div class="highlight-pycon notranslate"><div class="highlight"><pre>&gt;&gt;&gt; example.greet("Ada")
'Hello, Ada!'</pre></div></div>
    </dd>
  </dl>
  <section id="details">
    <h2>Details<a class="headerlink" href="#details">¶</a></h2>
    <p>Nested section text that explains the details of how greetings are produced.</p>
    <p>See <a href="functions.html#print">print()</a> and <a href="https://example.com/x.html">elsewhere</a>
       and <a href="../whatsnew/3.12.html">what's new</a>.</p>
  </section>
</section>
</div></body></html>
"""


@pytest.fixture
def page():
    return parse_page(HTML, URL)


def test_parse_page_sections(page):
    assert page.title == "example — An example module"
    titles = [s.title for s in page.sections]
    assert titles == ["example — An example module", "Details"]
    top = page.sections[0]
    assert top.paragraphs == ["This module provides example helpers for testing the parser."]
    assert top.code == ['import example\nexample.greet("world")']
    # nested section content is not duplicated into its parent
    assert all("Nested" not in p for p in top.paragraphs)


def test_parse_page_api_entries(page):
    (entry,) = page.api_entries
    assert entry.name == "example.greet"
    assert entry.kind == "function"
    assert entry.signature == "example.greet(name)"
    assert entry.description.startswith("Return a greeting for name")
    assert entry.code == [">>> example.greet(\"Ada\")\n'Hello, Ada!'"]


def test_extract_links_stays_in_scope():
    assert extract_links(HTML, URL) == ["https://docs.python.org/3/library/functions.html"]


def test_build_dataset(page):
    from dataclasses import asdict

    records = build([asdict(page)])
    tasks = {r["task"] for r in records}
    assert tasks == {"api", "concept", "code"}
    for r in records:
        assert set(r) == {"instruction", "input", "output", "source", "task"}
        assert r["output"].count("```") % 2 == 0
    api = next(r for r in records if r["task"] == "api")
    assert "`example.greet`" in api["instruction"]
    assert "```python" in api["output"]
    # deterministic
    assert build([asdict(page)]) == records


def test_split_by_page():
    recs = [{"source": f"https://d/{p}.html#{i}"} for p in "abcdefghij" for i in range(3)]
    train, val = split(recs, 0.2, seed=0)
    assert len(train) + len(val) == len(recs)
    val_pages = {r["source"].split("#")[0] for r in val}
    assert len(val_pages) == 2
    assert not val_pages & {r["source"].split("#")[0] for r in train}


def test_prompt_format():
    prompt = build_prompt("Sort a list")
    assert prompt.startswith("[INST] <<SYS>>") and prompt.endswith("Sort a list [/INST]")
    p, c = build_example("Q", "answer\n")
    assert p.endswith("[/INST]") and c == " answer"


def test_extract_code():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nand\n```\nx = 1\n```"
    assert extract_code(text) == "def f():\n    return 1"
    assert extract_code("  def g(): pass ") == "def g(): pass"


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------- reference solutions are correct


class ListNode:
    def __init__(self, val, next=None):
        self.val, self.next = val, next


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def _solution(pid):
    problem = next(p for p in load_problems() if p["id"] == pid)
    ns = {"ListNode": ListNode, "TreeNode": TreeNode}
    exec(problem["reference"], ns)
    return ns


CASES = {
    "contains-duplicate": ("contains_duplicate", [(([1, 2, 3, 1],), True), (([1, 2, 3],), False)]),
    "valid-anagram": ("is_anagram", [(("anagram", "nagaram"), True), (("rat", "car"), False)]),
    "two-sum": ("two_sum", [(([2, 7, 11, 15], 9), [0, 1]), (([3, 2, 4], 6), [1, 2])]),
    "top-k-frequent": ("top_k_frequent", [(([1, 1, 1, 2, 2, 3], 2), [1, 2])]),
    "valid-palindrome": ("is_palindrome", [(("A man, a plan, a canal: Panama",), True), (("race a car",), False)]),
    "container-with-most-water": ("max_area", [(([1, 8, 6, 2, 5, 4, 8, 3, 7],), 49)]),
    "best-time-to-buy-and-sell-stock": ("max_profit", [(([7, 1, 5, 3, 6, 4],), 5), (([7, 6, 4, 3, 1],), 0)]),
    "longest-substring-without-repeating": ("length_of_longest_substring", [(("abcabcbb",), 3), (("pwwkew",), 3), (("abba",), 2)]),
    "valid-parentheses": ("is_valid", [(("()[]{}",), True), (("(]",), False), (("(",), False)]),
    "binary-search": ("search", [(([-1, 0, 3, 5, 9, 12], 9), 4), (([-1, 0, 3, 5, 9, 12], 2), -1)]),
    "number-of-islands": ("num_islands", [(([list("11000"), list("11000"), list("00100"), list("00011")],), 3)]),
    "climbing-stairs": ("climb_stairs", [((2,), 2), ((5,), 8)]),
    "coin-change": ("coin_change", [(([1, 2, 5], 11), 3), (([2], 3), -1), (([1], 0), 0)]),
    "kth-largest-element": ("find_kth_largest", [(([3, 2, 1, 5, 6, 4], 2), 5), (([3, 2, 3, 1, 2, 4, 5, 5, 6], 4), 4)]),
    "merge-intervals": ("merge", [(([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]])]),
    "single-number": ("single_number", [(([4, 1, 2, 1, 2],), 4)]),
}


@pytest.mark.parametrize("pid", sorted(CASES))
def test_reference_solutions(pid):
    fn_name, cases = CASES[pid]
    fn = _solution(pid)[fn_name]
    for args, expected in cases:
        result = fn(*args)
        if pid == "top-k-frequent":
            result = sorted(result)
        assert result == expected, (pid, args)


def test_reference_group_anagrams():
    groups = _solution("group-anagrams")["group_anagrams"](["eat", "tea", "tan", "ate", "nat", "bat"])
    assert sorted(sorted(g) for g in groups) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]


def test_reference_subsets():
    assert sorted(_solution("subsets")["subsets"]([1, 2, 3])) == sorted(
        [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    )


def test_reference_linked_list_and_tree():
    head = ListNode(1, ListNode(2, ListNode(3)))
    rev = _solution("reverse-linked-list")["reverse_list"](head)
    vals = []
    while rev:
        vals.append(rev.val)
        rev = rev.next
    assert vals == [3, 2, 1]

    root = _solution("invert-binary-tree")["invert_tree"](TreeNode(2, TreeNode(1), TreeNode(3)))
    assert (root.left.val, root.right.val) == (3, 1)


def test_problem_file_schema():
    problems = load_problems()
    assert len({p["id"] for p in problems}) == len(problems) == 20
    for p in problems:
        assert {"id", "title", "category", "difficulty", "prompt", "reference"} <= set(p)
        compile(p["reference"], p["id"], "exec")
    json.dumps(problems)

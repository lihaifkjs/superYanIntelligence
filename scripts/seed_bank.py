"""Terminal 种子任务库：内置 30 条手写 smoke 任务 + 10 个参数化模板。

模板生成：任务参数随机化，check 命令由同一模板同构生成（检查零手写成本）。
所有模板参数来自安全字符池（小写字母/数字，无引号），可直接嵌入 shell。
"""

import random

TERMINAL_TASKS = [
    # --- 文件操作 ---
    {"id": "T01", "task": "Create a file named hello.txt containing exactly the line: hello world",
     "meta": {"check": "grep -qx 'hello world' hello.txt"}},
    {"id": "T02", "task": "Create the nested directory structure a/b/c",
     "meta": {"check": "test -d a/b/c"}},
    {"id": "T03", "task": "Write the numbers 1 to 10, one per line, into nums.txt",
     "meta": {"check": "test $(wc -l < nums.txt) -eq 10 && tail -1 nums.txt | grep -qx 10"}},
    {"id": "T04", "task": "Rename old.txt to new.txt",
     "meta": {"setup": "echo data > old.txt", "check": "test -f new.txt && ! test -f old.txt"}},
    {"id": "T05", "task": "Copy the file src.txt into the subdir backup/ (create the subdir if needed), keeping the same file name",
     "meta": {"setup": "echo important > src.txt", "check": "test -f backup/src.txt && diff -q src.txt backup/src.txt"}},
    {"id": "T06", "task": "Create 5 empty files named f1.txt through f5.txt",
     "meta": {"check": "test $(ls f*.txt | wc -l) -eq 5"}},
    {"id": "T07", "task": "Create a symbolic link link.txt pointing to target.txt",
     "meta": {"setup": "echo x > target.txt", "check": "test -L link.txt && test $(cat link.txt) = x"}},
    {"id": "T08", "task": "Delete all files ending in .tmp under the current directory, but keep every other file",
     "meta": {"setup": "mkdir -p d1/d2 && touch a.txt b.tmp d1/c.tmp d1/d2/e.tmp d1/keep.txt",
              "check": "! find . -name '*.tmp' | grep . && test -f a.txt && test -f d1/keep.txt"}},
    {"id": "T09", "task": "Make script.sh executable and run it to produce out.txt",
     "meta": {"setup": "printf '#!/bin/bash\\necho done > out.txt\\n' > script.sh",
              "check": "test -x script.sh && grep -qx done out.txt"}},
    # --- 文本处理 ---
    {"id": "T10", "task": "Sort the lines of data.txt numerically (smallest first) into sorted.txt",
     "meta": {"setup": "printf '10\\n2\\n1\\n' > data.txt", "check": "tr '\\n' ' ' < sorted.txt | grep -qx '1 2 10 '"}},
    {"id": "T11", "task": "Extract the second whitespace-separated column of table.txt into col2.txt",
     "meta": {"setup": "printf 'a 1 x\\nb 2 y\\nc 3 z\\n' > table.txt", "check": "tr '\\n' ' ' < col2.txt | grep -qx '1 2 3 '"}},
    {"id": "T12", "task": "Replace every occurrence of 'foo' with 'bar' in text.txt, in place",
     "meta": {"setup": "printf 'foo baz\\nfoo foo\\n' > text.txt", "check": "! grep -q foo text.txt && test $(grep -c bar text.txt) -eq 3"}},
    {"id": "T13", "task": "Count how many lines in log.txt contain the word ERROR and write just the number to count.txt",
     "meta": {"setup": "printf 'ERROR a\\nok\\nERROR b\\nERROR c\\n' > log.txt", "check": "grep -qx 3 count.txt"}},
    {"id": "T14", "task": "Write the number of distinct lines in dup.txt to unique.txt",
     "meta": {"setup": "printf 'x\\ny\\nx\\nz\\ny\\n' > dup.txt", "check": "grep -qx 3 unique.txt"}},
    {"id": "T15", "task": "Reverse the line order of input.txt into reversed.txt (last line first)",
     "meta": {"setup": "printf '1\\n2\\n3\\n' > input.txt", "check": "tr '\\n' ' ' < reversed.txt | grep -qx '3 2 1 '"}},
    {"id": "T16", "task": "Extract lines 2 through 4 (inclusive) of big.txt into middle.txt",
     "meta": {"setup": "seq 1 6 > big.txt", "check": "tr '\\n' ' ' < middle.txt | grep -qx '2 3 4 '"}},
    {"id": "T17", "task": "Concatenate one.txt and two.txt with a single blank line between them into merged.txt",
     "meta": {"setup": "printf 'A\\n' > one.txt && printf 'B\\n' > two.txt",
              "check": "tr '\\n' ' ' < merged.txt | grep -qx 'A B ' && test $(wc -l < merged.txt) -eq 3"}},
    {"id": "T18", "task": "Find the most frequent word in words.txt and write it alone to top.txt",
     "meta": {"setup": "printf 'cat dog cat bird dog cat\\n' > words.txt", "check": "grep -qx cat top.txt"}},
    {"id": "T19", "task": "Convert commas.csv (comma-separated) into a tab-separated file tabs.tsv",
     "meta": {"setup": "printf 'a,1\\nb,2\\n' > commas.csv", "check": "head -1 tabs.tsv | grep -qx $'a\\t1'"}},
    {"id": "T20", "task": "Write the lines of data.log that match the pattern 'user=[0-9]+' into matches.txt",
     "meta": {"setup": "printf 'user=12 ok\\nuser=x no\\nuser=7 ok\\n' > data.log",
              "check": "test $(wc -l < matches.txt) -eq 2 && grep -q 'user=12' matches.txt"}},
    {"id": "T21", "task": "Number every line of src.txt starting at 1 (line number, a space, then the line) into numbered.txt",
     "meta": {"setup": "printf 'aa\\nbb\\n' > src.txt",
              "check": "head -1 numbered.txt | grep -qx '1 aa' && tail -1 numbered.txt | grep -qx '2 bb'"}},
    # --- 查找与统计 ---
    {"id": "T22", "task": "List all .txt files under the current tree (recursively), sorted, one per line, into txtfiles.txt",
     "meta": {"setup": "mkdir -p u/v && touch z.txt u/a.txt u/v/b.txt u/c.md",
              "check": "tr '\\n' ' ' < txtfiles.txt | grep -qx './u/a.txt ./u/v/b.txt ./z.txt '"}},
    {"id": "T23", "task": "Write the total number of files (not directories) under the current tree to filecount.txt",
     "meta": {"setup": "mkdir -p m && touch m/1 m/2 top && touch m/3", "check": "grep -qx 4 filecount.txt"}},
    {"id": "T24", "task": "Find the largest file under the current tree and write its name (relative path) to biggest.txt",
     "meta": {"setup": "printf '12345' > large.bin && printf '1' > small.bin && mkdir -p sub && printf '123' > sub/mid.bin",
              "check": "grep -qx ./large.bin biggest.txt"}},
    # --- 压缩归档 ---
    {"id": "T25", "task": "Create a gzip-compressed archive archive.tar.gz of the folder pkg/",
     "meta": {"setup": "mkdir -p pkg && echo v1 > pkg/a.txt",
              "check": "tar -tzf archive.tar.gz | grep -q 'pkg/a.txt'"}},
    {"id": "T26", "task": "Compress note.txt into note.txt.gz and remove the original note.txt",
     "meta": {"setup": "echo secret > note.txt", "check": "test -f note.txt.gz && ! test -f note.txt && gzip -dc note.txt.gz | grep -qx secret"}},
    # --- git 基础 ---
    {"id": "T27", "task": "Initialize a git repository here, commit the existing file a.txt with message 'init'",
     "meta": {"setup": "echo a > a.txt",
              "check": "git log --oneline | grep -q init && test -z \"$(git status --porcelain)\""}},
    {"id": "T28", "task": "Create exactly 2 commits in this git repo: first commit b.txt with message 'add b', then modify b.txt and commit with message 'update b'",
     "meta": {"setup": "git init -q . && echo 1 > b.txt && git add b.txt && git -c user.email=t@t -c user.name=t commit -qm 'base'",
              "check": "test $(git log --oneline | wc -l) -eq 3 && git log --oneline | grep -q 'add b' && git log --oneline | grep -q 'update b'"}},
    {"id": "T29", "task": "In this git repo, create a new branch named feature and switch to it",
     "meta": {"setup": "git init -q . && echo x > f.txt && git add f.txt && git -c user.email=t@t -c user.name=t commit -qm init",
              "check": "git rev-parse --abbrev-ref HEAD | grep -qx feature"}},
    {"id": "T30", "task": "Write a bash script sum.sh that takes two integer arguments and prints their sum (nothing else). Verify it works.",
     "meta": {"check": "test \"$(bash sum.sh 3 4)\" = 7 && test \"$(bash sum.sh -1 1)\" = 0"}},
]


# ---------------------------------------------------------------- 模板化生成
# 手写 30 条只覆盖 smoke；规模化的任务由模板生成——任务参数随机化，
# check 命令由同一模板同构生成，检查零手写成本。
# 所有模板参数都来自安全字符池（小写字母/数字，无引号），可直接嵌入 shell。

_FILE_POOL = ["report", "notes", "alpha", "beta", "summary", "draft", "memo",
              "record", "ledger", "journal"]
_DIR_POOL = ["src", "data", "docs", "build", "dist", "assets", "config", "logs"]
_WORD_POOL = ["apple", "bravo", "delta", "echo", "foxtrot", "gold", "hotel",
              "india", "kilo", "lima"]


def _t_create_file(rng):
    name = rng.choice(_FILE_POOL)
    content = f"{rng.choice(_WORD_POOL)} {rng.choice(_WORD_POOL)}"
    return {
        "task": f"Create a file named {name}.txt containing exactly the line: {content}",
        "meta": {"check": f"grep -qx '{content}' {name}.txt"},
    }


def _t_nested_dirs(rng):
    a, b, c = rng.sample(_DIR_POOL, 3)
    return {
        "task": f"Create the nested directory structure {a}/{b}/{c}",
        "meta": {"check": f"test -d {a}/{b}/{c}"},
    }


def _t_write_numbers(rng):
    n = rng.randint(5, 20)
    fname = rng.choice(_FILE_POOL)
    return {
        "task": f"Write the numbers 1 to {n}, one per line, into {fname}.txt",
        "meta": {"check":
                 f"test $(wc -l < {fname}.txt) -eq {n} && "
                 f"tail -1 {fname}.txt | grep -qx {n}"},
    }


def _t_rename(rng):
    old, new = rng.sample(_FILE_POOL, 2)
    return {
        "task": f"Rename {old}.txt to {new}.txt",
        "meta": {"setup": f"echo data > {old}.txt",
                 "check": f"test -f {new}.txt && ! test -f {old}.txt"},
    }


def _t_count_word_lines(rng):
    word = rng.choice(_WORD_POOL)
    count = rng.randint(2, 6)
    others = rng.randint(1, 4)
    lines = [f"{word} item{i}" for i in range(count)] + \
            [f"other {i}" for i in range(others)]
    setup = "printf '" + "\\n".join(lines) + "\\n' > log.txt"
    return {
        "task": f"Count how many lines in log.txt contain the word '{word}' "
                f"and write just the number to result.txt",
        "meta": {"setup": setup, "check": f"grep -qx {count} result.txt"},
    }


def _t_sort_file(rng):
    nums = rng.sample(range(1, 100), rng.randint(5, 10))
    fname = rng.choice(_FILE_POOL)
    setup = "printf '" + "\\n".join(map(str, nums)) + f"\\n' > {fname}.txt"
    want = " ".join(map(str, sorted(nums))) + " "
    return {
        "task": f"Sort the lines of {fname}.txt numerically (smallest first) "
                f"into sorted.txt",
        "meta": {"setup": setup,
                 "check": f"tr '\\n' ' ' < sorted.txt | grep -qx '{want}'"},
    }


def _t_extract_column(rng):
    vals = rng.sample(range(1, 100), 4)
    keys = rng.sample(_WORD_POOL, 4)
    rows = [f"{k} {v} x" for k, v in zip(keys, vals)]
    setup = "printf '" + "\\n".join(rows) + "\\n' > table.txt"
    want = " ".join(map(str, vals)) + " "
    return {
        "task": "Extract the second whitespace-separated column of table.txt "
                "into col2.txt",
        "meta": {"setup": setup,
                 "check": f"tr '\\n' ' ' < col2.txt | grep -qx '{want}'"},
    }


def _t_sum_script(rng):
    a, b = rng.randint(2, 50), rng.randint(2, 50)
    c, d = rng.randint(-20, -1), rng.randint(2, 20)
    return {
        "task": "Write a bash script add.sh that takes two integer arguments "
                "and prints their sum (nothing else).",
        "meta": {"check":
                 f'test "$(bash add.sh {a} {b})" = "{a + b}" && '
                 f'test "$(bash add.sh {c} {d})" = "{c + d}"'},
    }


def _t_delete_ext(rng):
    ext = rng.choice(["tmp", "bak", "old"])
    return {
        "task": f"Delete all files ending in .{ext} under the current "
                f"directory, but keep every other file",
        "meta": {
            "setup": f"mkdir -p sub && touch keep.txt sub/keep2.txt "
                     f"a.{ext} b.{ext} sub/c.{ext}",
            "check": f"! find . -name '*.{ext}' | grep . && "
                     f"test -f keep.txt && test -f sub/keep2.txt",
        },
    }


def _t_git_commit(rng):
    msg = rng.choice(["initial", "setup", "first", "baseline"])
    return {
        "task": f"Initialize a git repository here and commit the existing "
                f"file main.txt with message '{msg}'",
        "meta": {
            "setup": "echo x > main.txt",
            "check": f"git log --oneline | grep -q {msg} && "
                     f'test -z "$(git status --porcelain)"',
        },
    }


TERMINAL_TEMPLATES = [
    ("create_file", _t_create_file),
    ("nested_dirs", _t_nested_dirs),
    ("write_numbers", _t_write_numbers),
    ("rename", _t_rename),
    ("count_word_lines", _t_count_word_lines),
    ("sort_file", _t_sort_file),
    ("extract_column", _t_extract_column),
    ("sum_script", _t_sum_script),
    ("delete_ext", _t_delete_ext),
    ("git_commit", _t_git_commit),
]


def build_generated_terminal(n: int, seed: int) -> list[dict]:
    """模板化生成 Terminal 任务；check 随任务同构产出。"""
    rng = random.Random(seed)
    seeds, seen = [], set()
    tries = 0
    while len(seeds) < n and tries < n * 20:
        tries += 1
        tname, fn = TERMINAL_TEMPLATES[rng.randrange(len(TERMINAL_TEMPLATES))]
        t = fn(rng)
        key = t["task"] + "|" + t["meta"].get("check", "")
        if key in seen:
            continue
        seen.add(key)
        t["meta"]["template"] = tname
        seeds.append(t)
    return [{
        "id": f"G{i+1:04d}",
        "expert": "terminal",
        "task": t["task"],
        "ground_truth": None,
        "meta": t["meta"],
    } for i, t in enumerate(seeds)]


def build_terminal_seeds() -> list[dict]:
    seeds = []
    for t in TERMINAL_TASKS:
        seeds.append({
            "id": t["id"],
            "expert": "terminal",
            "task": t["task"],
            "ground_truth": None,
            "meta": t.get("meta", {}),
        })
    return seeds



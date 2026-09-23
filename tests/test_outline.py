"""Tests for swival.outline — structural code outline tool."""

from swival.outline import SymbolSpan, outline, outline_files, symbol_spans
from swival.tools import OUTLINE_TOOL


def test_python_classes_functions_decorators(tmp_path):
    src = tmp_path / "example.py"
    src.write_text("""\
import os

@dataclass
class Foo(Base):
    x: int
    y: str = "hello"

    def method(self):
        pass

    @staticmethod
    def static_method():
        pass

@decorator
def top_func(a, b=1):
    pass

async def async_func():
    pass
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "class Foo(Base):" in result
    assert "@dataclass" in result
    assert "def method(self)" in result
    assert "@staticmethod" in result
    assert "def static_method()" in result
    assert "@decorator" in result
    assert "def top_func(a, b=...)" in result
    assert "async def async_func()" in result


def test_python_depth_1_top_level_only(tmp_path):
    src = tmp_path / "shallow.py"
    src.write_text("""\
class MyClass:
    def inner_method(self):
        pass

def top_level():
    pass
""")
    result = outline(str(src), str(tmp_path), depth=1, files_mode="all")
    assert "class MyClass:" in result
    assert "def top_level()" in result
    assert "inner_method" not in result


def test_python_depth_3_shows_nested(tmp_path):
    src = tmp_path / "deep.py"
    src.write_text("""\
class Outer:
    class Inner:
        def deep_method(self):
            pass
    def method(self):
        pass

def top():
    pass
""")
    result = outline(str(src), str(tmp_path), depth=3, files_mode="all")
    assert "class Outer:" in result
    assert "class Inner:" in result
    assert "def deep_method(self)" in result
    assert "def method(self)" in result
    assert "def top()" in result


def test_non_python_heuristic_js(tmp_path):
    src = tmp_path / "app.js"
    src.write_text("""\
// A comment
const API_URL = "http://example.com";

function greet(name) {
    return "hello " + name;
}

class Widget {
    constructor() {}
}

export function helper() {}
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "const API_URL" in result
    assert "function greet" in result
    assert "class Widget" in result
    assert "export function helper" in result
    assert "// A comment" not in result


def test_non_python_heuristic_go(tmp_path):
    src = tmp_path / "main.go"
    src.write_text("""\
package main

func main() {
    fmt.Println("hello")
}

type Server struct {
    addr string
}

func (s *Server) Start() error {
    return nil
}
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "package main" in result
    assert "func main()" in result
    assert "type Server struct" in result
    assert "func (s *Server) Start()" in result


def test_non_python_heuristic_rust(tmp_path):
    src = tmp_path / "lib.rs"
    src.write_text("""\
pub fn process(data: &[u8]) -> Result<(), Error> {
    Ok(())
}

struct Config {
    verbose: bool,
}

impl Config {
    fn new() -> Self {
        Config { verbose: false }
    }
}

enum Status {
    Ok,
    Err(String),
}
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "pub fn process" in result
    assert "struct Config" in result
    assert "impl Config" in result
    assert "fn new()" in result
    assert "enum Status" in result


def test_non_python_heuristic_zig(tmp_path):
    src = tmp_path / "main.zig"
    src.write_text("""\
const std = @import("std");

pub const Config = struct {
    verbose: bool = false,

    pub fn init() Config {
        return .{};
    }
};

pub fn process(data: []const u8) !void {
    const local_var = data.len;
    _ = local_var;
}

inline fn helper() void {}

test "process works" {
    try process("hello");
}
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    # Module-level const/pub const at depth 0
    assert "const std" in result
    assert "pub const Config" in result
    # pub fn inside struct at depth 1
    assert "pub fn init()" in result
    # Top-level pub fn
    assert "pub fn process" in result
    # inline fn
    assert "inline fn helper" in result
    # test blocks
    assert 'test "process works"' in result
    # Local variables inside function bodies must NOT appear
    assert "local_var" not in result


def test_non_python_heuristic_c(tmp_path):
    src = tmp_path / "main.c"
    src.write_text("""\
#include <stdio.h>

typedef struct {
    int x, y;
} Point;

enum Color { RED, GREEN, BLUE };

static int helper(int a, int b) {
    int local = a + b;
    return local;
}

int main(int argc, char *argv[]) {
    return 0;
}

void process(const char *data);
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "typedef struct" in result
    assert "enum Color" in result
    assert "static int helper" in result
    assert "int main" in result
    assert "void process" in result
    assert "local" not in result


def test_non_python_heuristic_cpp(tmp_path):
    src = tmp_path / "main.cpp"
    src.write_text("""\
#include <string>

namespace utils {

class Parser {
public:
    void parse();
};

}

int main() { return 0; }
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "namespace utils" in result
    assert "class Parser" in result
    assert "int main" in result


def test_non_python_heuristic_php(tmp_path):
    src = tmp_path / "app.php"
    src.write_text("""\
<?php
namespace App;

class Controller {
    public function index(): void {
        $x = 1;
    }

    private function validate(): bool {
        return true;
    }
}

interface Authenticatable {
    public function getId(): string;
}

trait Loggable {
    public function log(string $msg): void {}
}
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "namespace App" in result
    assert "class Controller" in result
    assert "public function index" in result
    assert "private function validate" in result
    assert "interface Authenticatable" in result
    assert "trait Loggable" in result
    assert "$x" not in result


def test_non_python_heuristic_swift(tmp_path):
    src = tmp_path / "main.swift"
    src.write_text("""\
struct Point {
    func distance(to other: Point) -> Double {
        let dx = 0.0
        return dx
    }
}

protocol Identifiable {
    func describe() -> String
}

class ViewModel {
    private func fetch() -> Data { Data() }
}
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "struct Point" in result
    assert "func distance" in result
    assert "protocol Identifiable" in result
    assert "private func fetch" in result
    assert "let dx" not in result


def test_non_python_heuristic_typescript(tmp_path):
    src = tmp_path / "app.ts"
    src.write_text("""\
interface Config {
    port: number;
}

type Handler = (req: Request) => void;

export class Server {
    public async start(): Promise<void> {
        const x = 1;
    }
}

abstract class Base {
    abstract handle(): void;
}

export const VERSION = '1.0';
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "interface Config" in result
    assert "type Handler" in result
    assert "export class Server" in result
    assert "abstract class Base" in result
    assert "export const VERSION" in result
    assert "const x" not in result


def test_empty_file(tmp_path):
    src = tmp_path / "empty.py"
    src.write_text("")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert result == "empty file"


def test_whitespace_only_file(tmp_path):
    src = tmp_path / "blank.py"
    src.write_text("   \n\n  \n")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert result == "empty file"


def test_binary_file(tmp_path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00\x01\x02\xff\xfe")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert result.startswith("error:")
    assert "binary" in result


def test_nonexistent_file(tmp_path):
    result = outline(
        str(tmp_path / "nope.py"), str(tmp_path), depth=2, files_mode="all"
    )
    assert result.startswith("error:")
    assert "not found" in result


def test_directory_empty(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    result = outline(str(d), str(tmp_path), files_mode="all")
    assert not result.startswith("error:")
    assert "empty directory" in result


def test_python_syntax_error_falls_back_to_heuristic(tmp_path):
    src = tmp_path / "broken.py"
    src.write_text("""\
def valid_func():
    pass

class Incomplete(
    # missing closing paren and colon

def another():
    pass
""")
    result = outline(str(src), str(tmp_path), depth=2, files_mode="all")
    assert "def valid_func" in result
    assert "def another" in result


def test_module_level_assignments_depth_1(tmp_path):
    src = tmp_path / "consts.py"
    src.write_text("""\
VERSION = "1.0"
DEBUG: bool = True
NAME: str

class Config:
    pass

def setup():
    x = 10
""")
    result = outline(str(src), str(tmp_path), depth=1, files_mode="all")
    assert "VERSION = ..." in result
    assert "DEBUG: bool = ..." in result
    assert "NAME: str" in result
    assert "class Config:" in result
    assert "def setup()" in result
    # x = 10 is inside a function, not module-level — should not appear
    assert "x = " not in result


def test_outline_depth_invalid_type(tmp_path):
    src = tmp_path / "f.py"
    src.write_text("class Foo: pass\n")
    result = outline(str(src), str(tmp_path), depth="bad", files_mode="all")
    assert result == "error: depth must be an integer"


def _make_py(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(body)
    return str(f)


def test_outline_files_batch(tmp_path):
    a = _make_py(tmp_path, "a.py", "class A:\n    def m(self): pass\n")
    b = _make_py(tmp_path, "b.py", "def top(): pass\n")
    result = outline_files(
        [{"file_path": a}, {"file_path": b}],
        str(tmp_path),
        files_mode="all",
    )
    assert "=== FILE:" in result
    assert "status: ok" in result
    assert "class A" in result
    assert "def top()" in result
    assert "files_succeeded: 2" in result
    assert "files_with_errors: 0" in result
    assert "batch_truncated: false" in result


def test_outline_files_per_file_depth(tmp_path):
    src = _make_py(tmp_path, "c.py", "class C:\n    def inner(self): pass\n")
    r1 = outline_files(
        [{"file_path": src, "depth": 1}],
        str(tmp_path),
        files_mode="all",
    )
    assert "class C" in r1
    assert "inner" not in r1

    r2 = outline_files(
        [{"file_path": src, "depth": 2}],
        str(tmp_path),
        files_mode="all",
    )
    assert "def inner" in r2


def test_outline_files_default_depth(tmp_path):
    src = _make_py(tmp_path, "d.py", "class D:\n    def method(self): pass\n")
    result = outline_files(
        [{"file_path": src}],
        str(tmp_path),
        default_depth=1,
        files_mode="all",
    )
    assert "class D" in result
    assert "method" not in result


def test_outline_files_with_error(tmp_path):
    good = _make_py(tmp_path, "good.py", "def ok(): pass\n")
    result = outline_files(
        [{"file_path": good}, {"file_path": str(tmp_path / "nope.py")}],
        str(tmp_path),
        files_mode="all",
    )
    assert "files_succeeded: 1" in result
    assert "files_with_errors: 1" in result
    assert "status: ok" in result
    assert "status: error" in result


def test_outline_files_empty_list(tmp_path):
    result = outline_files([], str(tmp_path), files_mode="all")
    assert result == "error: files list is empty"


def test_outline_files_too_many(tmp_path):
    files = [{"file_path": "x.py"}] * 21
    result = outline_files(files, str(tmp_path), files_mode="all")
    assert result.startswith("error: too many files requested")


def test_outline_files_string_entry(tmp_path):
    src = _make_py(tmp_path, "s.py", "def f(): pass\n")
    result = outline_files([src], str(tmp_path), files_mode="all")
    assert "files_succeeded: 1" in result
    assert "def f()" in result


def test_outline_files_missing_file_path(tmp_path):
    result = outline_files([{}], str(tmp_path), files_mode="all")
    assert "status: error" in result
    assert "missing file_path" in result
    assert "files_with_errors: 1" in result


def test_outline_files_invalid_entry_type(tmp_path):
    result = outline_files([42], str(tmp_path), files_mode="all")
    assert "status: error" in result
    assert "expected object or string, got int" in result
    assert "files_with_errors: 1" in result


def test_outline_files_header_counts(tmp_path):
    good = _make_py(tmp_path, "g.py", "def g(): pass\n")
    result = outline_files(
        [{"file_path": good}, {"file_path": str(tmp_path / "missing.py")}, 99],
        str(tmp_path),
        files_mode="all",
    )
    assert "files_succeeded: 1" in result
    assert "files_with_errors: 2" in result
    assert "batch_truncated: false" in result


def test_outline_files_budget_truncation_errors(tmp_path, monkeypatch):
    import swival.outline as outline_mod

    monkeypatch.setattr(outline_mod.tools, "MAX_OUTPUT_BYTES", 100)
    files = [{"file_path": str(tmp_path / f"miss{i}.py")} for i in range(10)]
    result = outline_files(files, str(tmp_path), files_mode="all")
    assert "batch_truncated: true" in result
    assert "[batch_truncated:" in result


def test_outline_files_budget_first_oversized_kept(tmp_path, monkeypatch):
    import swival.outline as outline_mod

    monkeypatch.setattr(outline_mod.tools, "MAX_OUTPUT_BYTES", 10)
    src = _make_py(tmp_path, "big.py", "def very_long_function_name(): pass\n")
    result = outline_files(
        [{"file_path": src}, {"file_path": src}],
        str(tmp_path),
        files_mode="all",
    )
    assert "files_succeeded: 1" in result
    assert "batch_truncated: true" in result
    assert "=== FILE:" in result
    assert "[batch_truncated: 1 file(s)" in result


def test_outline_files_budget_second_rejected_skip_count(tmp_path, monkeypatch):
    import swival.outline as outline_mod

    monkeypatch.setattr(outline_mod.tools, "MAX_OUTPUT_BYTES", 200)
    small = _make_py(tmp_path, "sm.py", "x = 1\n")
    big = _make_py(
        tmp_path, "bg.py", "\n".join(f"def func_{i}(): pass" for i in range(50)) + "\n"
    )
    result = outline_files(
        [{"file_path": small}, {"file_path": big}, {"file_path": small}],
        str(tmp_path),
        files_mode="all",
    )
    assert "files_succeeded: 1" in result
    assert "batch_truncated: true" in result
    assert "[batch_truncated: 2 file(s)" in result


def test_outline_tool_schema():
    props = OUTLINE_TOOL["function"]["parameters"]["properties"]
    assert "file_path" in props
    assert "files" in props
    assert "depth" in props
    assert props["files"]["maxItems"] == 20
    assert "required" not in OUTLINE_TOOL["function"]["parameters"]
    assert "directory" in props["file_path"]["description"]
    assert (
        "directory" in props["files"]["items"]["properties"]["file_path"]["description"]
    )
    assert "default" not in props["depth"]
    assert "1 for directory" in props["depth"]["description"]


def test_directory_survey_outlines_source_files(tmp_path):
    _make_py(tmp_path, "agent.py", "class Agent:\n    def run(self): pass\n")
    _make_py(tmp_path, "tools.py", "def dispatch(): pass\n")
    result = outline(str(tmp_path), str(tmp_path), files_mode="all")
    assert result.startswith("directory:")
    assert "=== FILE:" in result
    assert "class Agent" in result
    assert "def dispatch()" in result
    assert str(tmp_path) not in result


def test_directory_default_depth_is_one(tmp_path):
    _make_py(tmp_path, "m.py", "class Outer:\n    def inner(self): pass\n")
    implicit = outline(str(tmp_path), str(tmp_path), files_mode="all")
    assert "class Outer" in implicit
    assert "def inner" not in implicit

    explicit = outline(str(tmp_path), str(tmp_path), depth=2, files_mode="all")
    assert "def inner" in explicit


def test_file_default_depth_unchanged(tmp_path):
    src = _make_py(tmp_path, "f.py", "class C:\n    def method(self): pass\n")
    result = outline(src, str(tmp_path), files_mode="all")
    assert "def method" in result


def test_directory_only_subdirs_no_error(tmp_path):
    (tmp_path / "pkg").mkdir()
    _make_py(tmp_path / "pkg", "code.py", "def deep(): pass\n")
    result = outline(str(tmp_path), str(tmp_path), files_mode="all")
    assert not result.startswith("error:")
    assert "source_files: 0 selected" in result
    assert "pkg/" in result


def test_directory_excludes_noise(tmp_path):
    _make_py(tmp_path, "real.py", "def keep(): pass\n")
    (tmp_path / "package-lock.json").write_text('{"a": 1}\n')
    (tmp_path / "Cargo.lock").write_text("[[package]]\n")
    (tmp_path / "app.min.js").write_text("var x=1;\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "__pycache__").mkdir()
    result = outline(str(tmp_path), str(tmp_path), files_mode="all")
    assert "real.py" in result
    assert "package-lock.json" not in result
    assert "Cargo.lock" not in result
    assert "app.min.js" not in result
    assert "logo.png" not in result
    assert "__pycache__" not in result


def test_directory_caps_at_twenty(tmp_path):
    for i in range(25):
        _make_py(tmp_path, f"mod_{i:02d}.py", f"def f{i}(): pass\n")
    _make_py(tmp_path, "__init__.py", "VERSION = '1'\n")
    result = outline(str(tmp_path), str(tmp_path), files_mode="all")
    assert "omitted_over_cap:" in result
    assert result.count("=== FILE:") == 20
    assert "__init__.py" in result.split("=== FILE:")[1]


def test_directory_titles_are_relative(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    _make_py(sub, "agent.py", "def go(): pass\n")
    result = outline("pkg", str(tmp_path), files_mode="all")
    assert "=== FILE: pkg/agent.py ===" in result
    assert str(tmp_path) not in result


def test_directory_absolute_input_stays_relative(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    _make_py(sub, "agent.py", "def go(): pass\n")
    result = outline(str(sub), str(tmp_path), files_mode="all")
    assert "=== FILE: pkg/agent.py ===" in result
    assert "directory: pkg/" in result
    assert str(tmp_path) not in result


def test_directory_batch_item_resolves_kind_default(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    _make_py(sub, "code.py", "class K:\n    def inner(self): pass\n")
    sibling = _make_py(tmp_path, "lib.py", "class L:\n    def method(self): pass\n")
    result = outline_files(
        [{"file_path": "pkg"}, {"file_path": sibling}],
        str(tmp_path),
        files_mode="all",
    )
    assert "class K" in result and "inner" not in result
    assert "def method" in result


# ---------------------------------------------------------------------------
# symbol_spans
# ---------------------------------------------------------------------------


class TestSymbolSpansPython:
    def test_decorated_and_nested_functions(self):
        src = (
            "import os\n"  # 1
            "\n"  # 2
            '@app.route("/x")\n'  # 3
            "def handler(req):\n"  # 4
            "    def inner():\n"  # 5
            "        pass\n"  # 6
            "    return inner\n"  # 7
            "\n"  # 8
            "class Auth:\n"  # 9
            "    def check(self, tok):\n"  # 10
            "        return tok\n"  # 11
            "\n"  # 12
            "_private = 1\n"  # 13
            "def _helper():\n"  # 14
            "    pass\n"  # 15
        )
        spans = symbol_spans(src, "a.py")
        assert spans["handler"].start == 3  # decorator included
        assert spans["handler"].end == 7
        assert spans["handler"].render_end == 7
        assert spans["handler"].kind == "function"
        assert spans["handler"].truncated == ""
        assert spans["Auth"].start == 9
        assert spans["Auth"].end == 11
        assert spans["Auth"].kind == "class"
        # Underscore-prefixed top-level functions are kept.
        assert spans["_helper"].start == 14
        # Nested functions and methods are not standalone keys.
        assert "inner" not in spans
        assert "check" not in spans

    def test_defs_inside_top_level_if_and_try(self):
        src = (
            "if True:\n"
            "    def cond_fn():\n"
            "        pass\n"
            "try:\n"
            "    def try_fn():\n"
            "        pass\n"
            "except ImportError:\n"
            "    def fallback_fn():\n"
            "        pass\n"
        )
        spans = symbol_spans(src, "a.py")
        assert set(spans) == {"cond_fn", "try_fn", "fallback_fn"}


class TestSymbolSpansBraces:
    def test_c_functions_with_hostile_comments_and_strings(self):
        src = (
            "#include <stdio.h>\n"  # 1
            "\n"  # 2
            "/* a block\n"  # 3
            "   comment with } brace */\n"  # 4
            "static int helper(int a) {\n"  # 5
            '    char *s = "not a } brace";\n'  # 6
            "    if (a) {\n"  # 7
            "        return a + 1;\n"  # 8
            "    }\n"  # 9
            "    return 0;\n"  # 10
            "}\n"  # 11
            "\n"  # 12
            "int main(void)\n"  # 13
            "{\n"  # 14
            "    return helper(2);\n"  # 15
            "}\n"  # 16
            "\n"  # 17
            "int proto(int x);\n"  # 18
        )
        spans = symbol_spans(src, "a.c")
        assert (spans["helper"].start, spans["helper"].end) == (5, 11)
        # Allman-style brace on its own line.
        assert (spans["main"].start, spans["main"].end) == (13, 16)
        # Prototype ends at the semicolon.
        assert (spans["proto"].start, spans["proto"].end) == (18, 18)

    def test_go_function_and_receiver_method_scope(self):
        src = (
            "package main\n"  # 1
            "\n"  # 2
            "type Server struct {\n"  # 3
            "    addr string\n"  # 4
            "}\n"  # 5
            "\n"  # 6
            "func Handle(w int) {\n"  # 7
            "    println(w)\n"  # 8
            "}\n"  # 9
            "\n"  # 10
            "func (s *Server) method() {}\n"  # 11
        )
        spans = symbol_spans(src, "a.go")
        assert (spans["Handle"].start, spans["Handle"].end) == (7, 9)
        assert spans["Server"].kind == "class"
        # Documented v1 scope: receiver methods are not standalone keys.
        assert "method" not in spans

    def test_js_exported_arrow_function(self):
        src = (
            "export const parse = (raw) => {\n"
            '    return raw.split("{");\n'
            "};\n"
            "const MAX = 5;\n"
        )
        spans = symbol_spans(src, "a.js")
        assert (spans["parse"].start, spans["parse"].end) == (1, 3)
        assert spans["parse"].kind == "function"
        # Plain top-level values are out of v1 scope.
        assert "MAX" not in spans

    def test_unbalanced_braces_terminate_with_span_cap(self):
        src = "int broken(int a) {\n" + "    a += 1;\n" * 10
        spans = symbol_spans(src, "a.c")
        span = spans["broken"]
        assert span.truncated == "span-cap"
        assert span.end == span.render_end


class TestSymbolSpansFallbackAndCap:
    def test_indentation_fallback_on_unknown_language(self):
        src = (
            "def parse_config(raw)\n"  # 1
            "  data = decode(raw)\n"  # 2
            "  data\n"  # 3
            "end\n"  # 4
            "\n"  # 5
            "def other\n"  # 6
            "end\n"  # 7
        )
        spans = symbol_spans(src, "a.rb")
        assert (spans["parse_config"].start, spans["parse_config"].end) == (1, 4)

    def test_span_max_lines_caps_render_end_keeps_true_end(self):
        body = "".join(f"    x = {i}\n" for i in range(400))
        src = "def big():\n" + body + "\n\ndef after():\n    pass\n"
        spans = symbol_spans(src, "a.py")
        big = spans["big"]
        assert big.start == 1
        assert big.end == 401  # true end preserved
        assert big.render_end == 300
        assert big.truncated == "span-cap"
        assert spans["after"].truncated == ""

    def test_round_trip_dict(self):
        spans = symbol_spans("def f():\n    pass\n", "a.py")
        d = spans["f"].to_dict()
        assert d == {
            "start": 1,
            "end": 2,
            "render_end": 2,
            "kind": "function",
            "truncated": "",
        }
        assert SymbolSpan.from_dict(d) == spans["f"]

    def test_from_dict_defaults_for_absent_keys(self):
        span = SymbolSpan.from_dict({"start": 4, "end": 9})
        assert span.render_end == 9
        assert span.kind == "function"
        assert span.truncated == ""


# ---------------------------------------------------------------------------
# Line numbers agree with read_file, grep and edit_file
# ---------------------------------------------------------------------------

LS = "\N{LINE SEPARATOR}"
PS = "\N{PARAGRAPH SEPARATOR}"


def _outline_line(text: str, needle: str) -> int:
    for row in text.splitlines():
        if needle in row:
            return int(row.split()[0])
    raise AssertionError(f"{needle!r} not in outline:\n{text}")


def _display_line(source: str, needle: str) -> int:
    for i, row in enumerate(source.splitlines(), 1):
        if needle in row:
            return i
    raise AssertionError(needle)


class TestDisplayLineNumbers:
    def test_ordinary_file_keeps_ast_numbering(self, tmp_path):
        src = "import os\n\n\n@dec\ndef f(\n    a,\n):\n    return a\n"
        p = tmp_path / "a.py"
        p.write_text(src)
        result = outline(str(p), str(tmp_path))
        assert _outline_line(result, "@dec") == 4
        assert _outline_line(result, "def f") == 5
        assert (
            symbol_spans(src, "a.py")["f"].start,
            symbol_spans(src, "a.py")["f"].end,
        ) == (4, 8)

    def test_form_feed_line_before_def(self, tmp_path):
        src = "import os\n\x0c\ndef f():\n    return 1\n"
        p = tmp_path / "a.py"
        p.write_text(src)
        result = outline(str(p), str(tmp_path))
        assert _outline_line(result, "def f") == _display_line(src, "def f") == 4
        span = symbol_spans(src, "a.py")["f"]
        assert (span.start, span.end) == (4, 5)

    def test_form_feed_on_the_declaration_line(self, tmp_path):
        src = "x = 1\n\x0cdef f():\n    return 1\n"
        p = tmp_path / "a.py"
        p.write_text(src)
        assert _outline_line(outline(str(p), str(tmp_path)), "def f") == 3
        span = symbol_spans(src, "a.py")["f"]
        assert (span.start, span.end) == (3, 4)

    def test_separators_in_docstring_shift_later_declarations(self, tmp_path):
        src = (
            f'"""Module{LS}doc{PS}with\x85separators\x1c."""\n'
            "class A:\n"
            "    def m(self):\n"
            "        return 1\n"
        )
        p = tmp_path / "a.py"
        p.write_text(src)
        result = outline(str(p), str(tmp_path))
        assert _outline_line(result, "class A") == _display_line(src, "class A") == 6
        assert _outline_line(result, "def m") == 7
        span = symbol_spans(src, "a.py")["A"]
        assert (span.start, span.end) == (6, 8)

    def test_separator_inside_body_moves_only_the_end(self):
        src = 'def f():\n    s = "a\x0bb"\n    return s\n\ndef g():\n    pass\n'
        spans = symbol_spans(src, "a.py")
        assert (spans["f"].start, spans["f"].end) == (1, 4)
        assert (spans["g"].start, spans["g"].end) == (6, 7)

    def test_non_ascii_before_the_column_offset(self, tmp_path):
        src = f'x = "ééééé{LS}"; y = 2\nz = 3\n'
        p = tmp_path / "a.py"
        p.write_text(src)
        result = outline(str(p), str(tmp_path))
        assert _outline_line(result, "y = ...") == 2
        assert _outline_line(result, "z = ...") == 3

    def test_decorated_multiline_signature_nested_method(self, tmp_path):
        src = (
            "# intro\x0c\n"
            "class Outer:\n"
            "    class Inner:\n"
            "        @staticmethod\n"
            "        @other(\n"
            f'            "arg{LS}"\n'
            "        )\n"
            "        def method(\n"
            "            a,\n"
            "        ):\n"
            "            return a\n"
        )
        p = tmp_path / "a.py"
        p.write_text(src)
        result = outline(str(p), str(tmp_path), depth=3)
        assert _outline_line(result, "class Inner") == 4
        assert _outline_line(result, "@staticmethod") == 5
        assert _outline_line(result, "def method") == _display_line(src, "def method")
        span = symbol_spans(src, "a.py")["Outer"]
        assert (span.start, span.end) == (3, 13)

    def test_long_function_keeps_true_end_past_the_render_cap(self):
        body = "".join(f"    x = {i}  # \x0c\n" for i in range(350))
        src = "def big():\n" + body
        span = symbol_spans(src, "a.py")["big"]
        assert span.start == 1
        assert "x = 349" in src.splitlines()[span.end - 1]
        assert span.render_end == 300
        assert span.truncated == "span-cap"

    def test_spans_read_back_through_read_file(self, tmp_path):
        from swival.tools import _read_file

        src = (
            f'"""Doc{LS}string."""\n\x0c\n'
            "@decorator\n"
            "def target(a,\n"
            "           b):\n"
            f'    note = "x{PS}y"\n'
            "    return a + b\n"
            "\n"
            "def neighbour():\n"
            "    pass\n"
        )
        (tmp_path / "m.py").write_text(src)
        span = symbol_spans(src, "m.py")["target"]
        shown = _read_file(
            "m.py", str(tmp_path), offset=span.start, limit=span.end - span.start + 1
        )
        assert "@decorator" in shown
        assert "return a + b" in shown
        assert "neighbour" not in shown

    def test_heuristic_spans_count_form_feed_lines(self):
        src = (
            "/* intro */\n\x0c\n"
            "int helper(int x)\n{\n  return x;\n}\n\n"
            "int main(void)\n{\n  return helper(1);\n}\n"
        )
        spans = symbol_spans(src, "a.c")
        lines = src.splitlines()
        assert lines[spans["helper"].start - 1] == "int helper(int x)"
        assert lines[spans["helper"].end - 1] == "}"
        assert (spans["main"].start, spans["main"].end) == (9, 12)

    def test_heuristic_spans_follow_lone_carriage_returns(self):
        src = "int a(void)\r{\r  return 1;\r}\rint b(void)\r{\r}\r"
        spans = symbol_spans(src, "a.c")
        assert (spans["a"].start, spans["a"].end) == (1, 4)
        assert (spans["b"].start, spans["b"].end) == (5, 7)

    def test_generated_sources_match_splitlines(self):
        import ast
        import random

        from swival.outline import _AstLines

        rng = random.Random(1234)
        separators = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", LS, PS]
        for case in range(300):
            parts: list[str] = []
            expected: dict[str, tuple[int, int]] = {}

            def line_at(offset: int) -> int:
                return len(("".join(parts)[:offset] + "x").splitlines())

            for i in range(rng.randint(1, 6)):
                sep = rng.choice(separators)
                kind = rng.randrange(4)
                if kind == 0:
                    parts.append(f"# c{sep}é{rng.choice(separators)}\n")
                elif kind == 1:
                    parts.append(f'"""d{sep}ö"""\n')
                elif kind == 2:
                    parts.append("\x0c\n")
                name = f"f{i}"
                if rng.random() < 0.5:
                    start = len("".join(parts))
                    parts.append(f"@dec{i}\n")
                else:
                    start = None
                if rng.random() < 0.3:
                    parts.append("\x0c")
                def_at = len("".join(parts))
                parts.append(
                    f"def {name}(a,{sep if rng.random() < 0.2 else ''}\n        b):\n"
                )
                parts.append(
                    f'    s = "é{sep}x"\n' if rng.random() < 0.5 else "    s = 1\n"
                )
                parts.append("    return s")
                end_at = len("".join(parts)) - 1
                parts.append("\n")
                first = line_at(def_at if start is None else start)
                expected[name] = (first, line_at(end_at), line_at(def_at))
            src = "".join(parts)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            spans = symbol_spans(src, "g.py")
            pos = _AstLines(src)
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                first, last, def_line = expected[node.name]
                assert pos.start(node) == def_line, (case, src)
                assert (spans[node.name].start, spans[node.name].end) == (
                    first,
                    last,
                ), (
                    case,
                    src,
                )


def test_form_feed_between_at_and_decorator_expression(tmp_path):
    src = "@\x0cdecorator\ndef f():\n    pass\n"
    p = tmp_path / "a.py"
    p.write_text(src)
    result = outline(str(p), str(tmp_path))
    assert _outline_line(result, "@decorator") == 1
    assert _outline_line(result, "def f") == 3
    span = symbol_spans(src, "a.py")["f"]
    assert (span.start, span.end) == (1, 4)


def test_at_sign_in_a_comment_inside_a_decorator(tmp_path):
    src = "\x0c\n@(\n    # @ unrelated comment\n    decorator\n)\ndef f():\n    pass\n"
    p = tmp_path / "a.py"
    p.write_text(src)
    result = outline(str(p), str(tmp_path))
    assert _outline_line(result, "@decorator") == 3
    assert symbol_spans(src, "a.py")["f"].start == 3

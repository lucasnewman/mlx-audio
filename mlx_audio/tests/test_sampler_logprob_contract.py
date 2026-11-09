"""make_sampler() samplers take log-probabilities, not raw logits.

``apply_top_p`` assumes ``sum(exp(x)) == 1``; feeding raw
logits leaves ``exp`` unnormalized and silently disables tail pruning while
argmax/top_k/min_p keep working, so the regression is invisible without this scan.

The scan walks model source files that import ``make_sampler`` from
``mlx_audio``, collects names bound to ``make_sampler(...)`` calls, and flags
calls like ``sampler(logits)`` whose argument expression mentions a ``*logit*``
identifier without any normalization (``log_softmax``/``logsumexp``).

Known blind spots: samplers built via ``partial``, rebound names,
containers, parameter defaults, or inline ``make_sampler(...)(x)`` calls are
not tracked, and arguments named without ``logit`` (e.g. ``scores``) are not
recognized as logit-valued.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mlx_audio"

# Presence of one of these in an expression means logits were normalized
# into log-probabilities before reaching the sampler.
NORMALIZERS = ("log_softmax", "logsumexp")

# Allowed "path:call" pairs (unparse() output, so line shifts do not break
# them). The scan fails if an entry stops matching so the list cannot go
# stale; remove the entry when the call site is fixed or deleted.
# Currently empty: every make_sampler caller is expected to normalize.
KNOWN_VIOLATIONS = set()


def _model_sources():
    for path in sorted(SOURCE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if "/tests/" in rel:
            continue
        yield rel, path


def _make_sampler_names(tree):
    """Local names bound to mlx_audio's make_sampler, aliases included."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "mlx_audio"
        ):
            for alias in node.names:
                if alias.name == "make_sampler":
                    names.add(alias.asname or alias.name)
    return names


def _is_make_sampler_call(node, names) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in names
    if isinstance(func, ast.Attribute):
        return func.attr == "make_sampler"
    return False


def _binds_sampler(value, names) -> bool:
    """Whether an assigned expression can yield a make_sampler() sampler."""
    if _is_make_sampler_call(value, names):
        return True
    if isinstance(value, ast.BoolOp):  # sampler or make_sampler(...)
        return any(_is_make_sampler_call(operand, names) for operand in value.values)
    if isinstance(value, ast.IfExp):
        return _is_make_sampler_call(value.body, names) or _is_make_sampler_call(
            value.orelse, names
        )
    return False


def _target_names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Attribute):
        # Only self.X bindings are tracked, and only self.X calls can match
        # them below; an unrelated model.sampler(...) call is not flagged.
        if isinstance(target.value, ast.Name) and target.value.id == "self":
            yield target.attr
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _expr_names(node):
    """Every Name/Attribute identifier appearing inside an expression."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Attribute):
            yield child.attr


def _is_logit_valued(arg, normalized_names) -> bool:
    names = list(_expr_names(arg))
    if any(name in NORMALIZERS for name in names):
        return False
    return any(
        "logit" in name and "logprob" not in name and name not in normalized_names
        for name in names
    )


def _collect_bindings(tree, names):
    """Names bound to make_sampler() output, and names holding log-probs."""
    sampler_names, normalized_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value, targets = node.value, [node.target]
        else:
            continue
        if _binds_sampler(value, names):
            for target in targets:
                sampler_names.update(_target_names(target))
        if any(name in NORMALIZERS for name in _expr_names(value)):
            for target in targets:
                normalized_names.update(_target_names(target))
    return sampler_names, normalized_names


def _sampler_calls_with_logits(tree, sampler_names, normalized_names):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called = func.id
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            called = func.attr
        else:
            continue
        if called not in sampler_names:
            continue
        args = list(node.args) + [kw.value for kw in node.keywords]
        if any(_is_logit_valued(arg, normalized_names) for arg in args):
            yield node


def test_make_sampler_callers_pass_logprobs_not_logits():
    flagged = {}
    for rel, path in _model_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = _make_sampler_names(tree)
        if not names:
            continue
        sampler_names, normalized_names = _collect_bindings(tree, names)
        for node in _sampler_calls_with_logits(tree, sampler_names, normalized_names):
            flagged[f"{rel}:{ast.unparse(node)}"] = node.lineno

    unexpected = sorted(flagged.keys() - KNOWN_VIOLATIONS)
    stale = sorted(KNOWN_VIOLATIONS - flagged.keys())
    assert not unexpected and not stale, (
        "make_sampler() samplers take log-probabilities; passing raw logits "
        "silently disables top_p. Normalize with log_softmax or "
        "logits - mx.logsumexp(logits, axis=-1, keepdims=True) first.\n"
        f"New violations {[(key, flagged[key]) for key in unexpected]}\n"
        f"Stale KNOWN_VIOLATIONS entries (site fixed, remove them): {stale}"
    )

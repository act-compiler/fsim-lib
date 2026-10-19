# The nki suite: every source's case list, checked against the folders.
# A case is listed by gen/<source>/cases.py and lives in <name>/ here.
# A bench names a case, a directory of cases, or a name prefix.
#   suite.py [bench...]    prints the cases the benches resolve to
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(ROOT, 'gen')
PREFIX = 'benchmarks/nki/'


def load():
    # Every source's cases, sources in name order.
    cases = []
    for source in sorted(os.listdir(GEN)):
        if not os.path.isdir(os.path.join(GEN, source)):
            continue
        path = os.path.join(GEN, source, 'cases.py')
        spec = importlib.util.spec_from_file_location(f'cases_{source}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for case in module.CASES:
            if not case['name'].startswith(source + '/'):
                sys.exit(f'{path}: case {case["name"]} is not under {source}/')
            cases.append(case)
    check(cases)
    return cases


def check(cases):
    # Lists and folders agree: each listed case has a .gitkeep, each folder a case.
    names = [c['name'] for c in cases]
    errors = [f'{n}: listed twice' for n in sorted(set(names)) if names.count(n) > 1]
    errors += [f'{n}: listed, but no {n}/.gitkeep' for n in names if not gitkept(n)]
    for d, dirs, files in os.walk(ROOT):
        rel = os.path.relpath(d, ROOT)
        if rel == '.':
            dirs.remove('gen')
        elif ('.gitkeep' in files or 'kernel.neff' in files) and rel not in names:
            errors.append(f'{rel}: a case folder no cases.py lists')
    if errors:
        sys.exit('case lists and folders disagree:\n  ' + '\n  '.join(errors))


def gitkept(name):
    return os.path.isfile(os.path.join(ROOT, name, '.gitkeep'))


def matches(cases, bench):
    # Exact name, then name prefix.
    exact = [c for c in cases if c['name'] == bench]
    if exact:
        return exact
    return [c for c in cases if c['name'].startswith(bench)]


def resolve(cases, benches):
    # The cases any bench matches, in suite order; all when none.
    if not benches:
        return cases
    picked = set()
    unmatched = []
    for bench in benches:
        hits = matches(cases, bench.removeprefix('./').removeprefix(PREFIX).rstrip('/'))
        picked.update(c['name'] for c in hits)
        if not hits:
            unmatched.append(bench)
    if unmatched:
        sys.exit('no case matches: ' + ' '.join(unmatched))
    return [c for c in cases if c['name'] in picked]


if __name__ == '__main__':
    for case in resolve(load(), sys.argv[1:]):
        print(case['name'])

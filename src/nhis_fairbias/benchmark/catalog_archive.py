"""Read-only relocation of frozen catalog evidence without rewriting identities.

The JSON retains the original server paths. Only filesystem reads are mapped
to explicit local archives; every original digest is still checked by the
reporting validator. This adapter never selects files by modification time,
filename search, or a successful digest among alternate copies.
"""
from pathlib import Path, PurePosixPath

from . import result_catalog as catalog


class ArchiveCatalog:
    def __init__(self, mappings, *, local_input_roots, output_dir):
        catalog._require(isinstance(mappings, list) and mappings, 'MISSING_ARCHIVE_MAPPINGS')
        self.mappings, prefixes = [], set()
        for item in mappings:
            catalog._require(isinstance(item, dict) and set(item) == {'recorded_prefix', 'local_root'},
                             'INVALID_ARCHIVE_MAPPING')
            prefix = self._absolute(item['recorded_prefix'])
            catalog._require(str(prefix) not in prefixes, 'DUPLICATE_ARCHIVE_PREFIX')
            prefixes.add(str(prefix))
            target = self._absolute(item['local_root']).resolve()
            catalog._require(target.is_dir(), 'MISSING_ARCHIVE_ROOT')
            self.mappings.append((prefix, target))
        self.mappings.sort(key=lambda pair: len(pair[0].parts), reverse=True)
        self.local_input_roots = [self._absolute(str(root)).resolve() for root in local_input_roots]
        catalog._require(all(root.is_dir() for root in self.local_input_roots), 'MISSING_LOCAL_INPUT_ROOT')
        self.output_dir = self._absolute(str(output_dir)).resolve()
        self.protect_output(self.output_dir)

    @staticmethod
    def _absolute(value):
        catalog._require(isinstance(value, str), 'INVALID_ARCHIVE_PATH')
        raw = PurePosixPath(value)
        catalog._require(raw.is_absolute() and '..' not in raw.parts and raw.parent != raw,
                         'INVALID_ARCHIVE_PATH')
        return Path(value)

    def physical(self, value):
        logical = self._absolute(str(value))
        for prefix, target in self.mappings:
            if logical.is_relative_to(prefix):
                result = (target / logical.relative_to(prefix)).resolve()
                catalog._require(result.is_relative_to(target), 'ARCHIVE_PATH_ESCAPE')
                return result
        for root in [*self.local_input_roots, self.output_dir]:
            if logical.is_relative_to(root):
                result = logical.resolve()
                catalog._require(result.is_relative_to(root), 'ARCHIVE_PATH_ESCAPE')
                return result
        raise catalog.CatalogError('UNMAPPED_ARCHIVE_PATH')

    def protect_output(self, output):
        out = self._absolute(str(output)).resolve()
        catalog._require(out == self.output_dir, 'ARCHIVE_OUTPUT_MISMATCH')
        roots = [target for _, target in self.mappings] + self.local_input_roots
        catalog._require(all(not out.is_relative_to(root) and not root.is_relative_to(out) for root in roots),
                         'OUTPUT_INSIDE_EVIDENCE')
        # Prevent a logical server path from being used as a local output.
        catalog._require(all(not out.is_relative_to(prefix) for prefix, _ in self.mappings),
                         'OUTPUT_INSIDE_EVIDENCE')

    def read_metadata(self, path, fields=None):
        return catalog.read_metadata(self.physical(path), fields)

    def __getattr__(self, name):
        # Constants and identity hashing are exactly the existing implementation.
        return getattr(catalog, name)

    def _Files(self, roots, relocations):
        return _ArchiveFiles(roots, relocations, self)


class _ArchiveFiles(catalog._Files):
    def __init__(self, roots, relocations, archive):
        # Retain logical paths for all identity/path equality tests. Only root
        # existence and byte reads use physical paths. ref/child/recorded and
        # verify/source_manifest are inherited without changing their contracts.
        catalog._require(isinstance(roots, dict) and roots, 'MISSING_ROOTS')
        self.archive, self.roots, self.hashes = archive, {}, {}
        for name, value in roots.items():
            catalog._require(isinstance(name, str) and catalog.LABEL.fullmatch(name), 'INVALID_ROOT_ID')
            path = archive._absolute(value)
            catalog._require(archive.physical(path).is_dir(), 'INVALID_ROOT_DIRECTORY')
            self.roots[name] = path.resolve()
        self.relocations, seen = [], set()
        for item in relocations:
            prefix = PurePosixPath(item['recorded_prefix'])
            catalog._require(prefix.is_absolute() and '..' not in prefix.parts and str(prefix) not in seen,
                             'INVALID_OR_DUPLICATE_RELOCATION')
            seen.add(str(prefix))
            self.relocations.append((prefix, self.ref(item['target'])))
        self.relocations.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    def sha(self, path):
        return super().sha(self.archive.physical(path))

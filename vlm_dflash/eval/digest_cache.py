"""Reuse SHA-256 only while inode, size, nanosecond mtime and ctime match.

The cache is an operational I/O optimization, not an adversarial integrity boundary.
A full read is performed on the first observation or whenever metadata changes.
"""
import hashlib
import os
from pathlib import Path
import sqlite3
import time


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _hash_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def cached_sha256(path, database):
    path = Path(path)
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with path.open('rb') as stream:
        key = _identity(os.fstat(stream.fileno()))
        if _identity(path.stat()) != key:
            raise ValueError(f'File changed before hashing: {path}')
        # Linux may assign equal inode timestamps to writes in the same clock tick.
        # Let the inode clock advance before checking/reading so writes during the
        # check cannot hide behind the initial timestamp on the local ext4 filesystems.
        time.sleep(0.02)
        with sqlite3.connect(database, timeout=60) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS digests (dev INTEGER, ino INTEGER, size INTEGER, mtime INTEGER, ctime INTEGER, sha TEXT NOT NULL, PRIMARY KEY(dev,ino,size,mtime,ctime))')
            row = conn.execute('SELECT sha FROM digests WHERE dev=? AND ino=? AND size=? AND mtime=? AND ctime=?',key).fetchone()
            if row is None:
                value = _hash_stream(stream)
            else:
                value = row[0]
                if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Invalid stored SHA-256')
            if _identity(os.fstat(stream.fileno())) != key or _identity(path.stat()) != key:
                raise ValueError(f'File changed during integrity check: {path}')
            if row is None:
                conn.execute('INSERT OR REPLACE INTO digests VALUES (?,?,?,?,?,?)', (*key,value))
        return value

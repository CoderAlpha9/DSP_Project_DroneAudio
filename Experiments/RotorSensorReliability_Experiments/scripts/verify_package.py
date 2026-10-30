"""Verify the packaged files before execution (new run files are allowed)."""
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];count=0
for line in (root/'CHECKSUMS_SHA256.txt').read_text().splitlines():
 digest,name=line.split('  ',1);p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('Checksum mismatch: '+name)
 count+=1
print('PASS:',count,'packaged file checksums')

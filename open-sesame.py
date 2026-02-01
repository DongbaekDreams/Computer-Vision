from pathlib import Path
import wfdb

base = Path(r"C:\Users\jacob\Downloads")

print("Base:", base)
print("Exists:", base.exists())

heas = sorted(base.glob("*.hea"))
mats = sorted(base.glob("*.mat"))

print(f".hea files: {len(heas)}")
print(f".mat files: {len(mats)}")

records = sorted({p.stem for p in heas} & {p.stem for p in mats})
print(f"records with both .hea and .mat: {len(records)}")
print("first 20 records:", records[:20])

if not records:
    raise SystemExit("No records found (need matching .hea + .mat in the same folder).")

name = records[0]
rec = wfdb.rdrecord(str(base / name))

print("Loaded:", name)
print("fs:", rec.fs)
print("sig_name:", rec.sig_name)
print("p_signal shape:", rec.p_signal.shape)

# quick peek at first 5 samples, first up to 5 channels
rows = min(5, rec.p_signal.shape[0])
cols = min(5, rec.p_signal.shape[1])
print("first samples:\n", rec.p_signal[:rows, :cols])



from pathlib import Path

hea = Path(r"C:\Users\jacob\Downloads\JS00001.hea")
print(hea.read_text())

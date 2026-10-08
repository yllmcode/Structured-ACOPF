# ACOPF Structured Serialization for LLM

Each system contains 9,000 training records and 1,000 test records. The saved
records are already serialized.

## Representation implemented here

The input is a deterministic, physically structured sequence with explicit
component blocks, topology-aware canonical ordering, per-unit values,
four-decimal numerical precision, and aligned component identifiers. The
immutable MATPOWER case named in the input supplies complete static topology,
branch parameters, equipment limits, and generation costs. Each target contains
the optimal objective, generator active/reactive power, bus voltage magnitude
and angle.

Every JSONL line contains:

- `instruction`: the ACOPF prediction instruction;
- `input`: the structured grid and operating-state sequence;
- `output`: the aligned IPOPT ACOPF target sequence;
- `method`: `proposed`;
- `case`: `case9`, `case30`, or `case118`;
- `source_index`: the immutable source-row identifier.


## Recreate serialized records

Install the small runtime:

```powershell
python -m pip install -r requirements.txt
```

Serialize an NPZ source containing `pd_pu`, `qd_pu`, `pg_pu`, `qg_pu`,
`vm_pu`, `va_rad`, `objective`, and optionally `source_index`:

```powershell
python -m serialization.serialize_samples `
  --case cases/case118.m `
  --data path/to/case118_train.npz `
  --output outputs/case118_train.jsonl
```


## Verify the release

From this directory run:

```powershell
python -m serialization.verify_release
```

The verifier checks the three original cases, all 30,000 serialized records,
file hashes, split sizes and disjointness, strict target completeness, component
dimensions, branch-terminal states, metadata, and the documented IEEE 118 token
count.

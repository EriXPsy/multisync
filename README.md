# MultiSync

Dynamic process analysis and visualization for multimodal interpersonal synchrony research.

## Architecture

MultiSync follows a **Python Core + React Viewer** architecture, separating computation from visualization:

### multisync-core (Python)

The computational engine. All statistical analysis, signal processing, and data transformation happen here.

```
multisync-core/
├── multisync/
│   ├── dataset.py          # SynchronyDataset: multi-Hz alignment, interpolation, normalization
│   ├── core.py             # Dyad + DynamicAnalyzer: 4-line API (load → context → analyze → export)
│   ├── cascade.py          # CCF with Hanning window + PRTF phase randomization surrogates
│   ├── dynamic_features.py # Vectorized WCC (stride_tricks) + 10 Gordon dynamic features
│   ├── prediction.py       # sklearn TimeSeriesSplit+gap + LODO + leakage audit
│   ├── synthetic.py        # Ground truth generator (Gaussian bursts, configurable noise/lag/dropout)
│   ├── io.py               # CSV reader + Viewer JSON Schema (zero frontend computation)
│   ├── normalization.py    # Within-dyad Z-score normalization (mandatory preprocessing)
│   └── cli.py              # CLI: python -m multisync analyze / python -m multisync demo
├── tests/
└── pyproject.toml
```

Install and run:

```bash
pip install -e "multisync-core[dev]"
python -m multisync demo
```

### MultiSync Web (React)

A pure visualization viewer. Reads `results.json` from multisync-core — no statistical computation in the browser.

Built with Vite + React + TypeScript + Tailwind + shadcn/ui + Recharts.

```bash
npm install
npm run dev
```

## Research Context

MultiSync is designed for researchers studying interpersonal synchrony across neural, behavioral, and physiological modalities. The theoretical framework builds on Gordon et al. (2024) and focuses on:

- **Dynamic process analysis**: onset latency, build-up rate, cascade patterns (not static feature comparison)
- **Cross-modal cascading**: which modality synchronizes first and guides others ("the overture signal")
- **Prediction windows**: identifying early synchrony patterns that predict full multimodal coupling

## License

MIT

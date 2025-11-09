# PRAK - Multi-Object Tracking Library

Rust port of MATLAB multisensor-lmb-filters library.

## Status

**Current Progress:** Phase 3 Complete (50%)
- ✅ Phase 1: Foundation (316 lines)
- ✅ Phase 2: Core Utilities (2,659 lines)  
- ✅ Phase 3: LMB Filter (1,379 lines)
- ⏳ Phase 4: LMBM Filter (in progress)
- ⏸ Phase 5-9: Multi-sensor, tests, demos, docs

**Total:** 4,327 lines | 44 tests passing

## Implemented

### LMB Filter (Complete)
- Cardinality estimation (ESF, MAP)
- Prediction step
- Association matrices (LBP, Gibbs, Murty's)  
- Posterior computation
- Main filter pipeline

### Core Utilities
- Linear algebra (Kalman, Gaussian PDF, log-sum-exp)
- Data association (Hungarian, LBP, Gibbs, Murty's)
- OSPA metrics (Euclidean & Hellinger)
- Gaussian mixture utilities

## Build

```bash
cargo build
cargo test
```


## 2026-02-16: Performance Run
- **Report**: `docs/reports/performance_2026-02-16_213339.md`
- **Status**: FAILED
- **System**: 13th Gen Intel(R) Core(TM) i5-13500
\n## 2026-02-20: LU Performance Run\n- **Report**: `docs/reports/performance_2026-02-20_214924.md`\n- **Status**: PASSED (LU Max Ratio < 1.08x)\n- **System**: 13th Gen Intel(R) Core(TM) i5-13500\n- **Notes**: Optimization replacing checked `unwrap()` with `get_unchecked` in scalar fallback loops improved maximum LU deviation from 1.13x -> 1.08x.

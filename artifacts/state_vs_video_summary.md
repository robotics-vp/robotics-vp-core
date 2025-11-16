# State vs Video Mode Comparison

**Date**: 2025-11-12 22:02:21

**Episodes**: 172 (each mode)

**Analysis Window**: Last 20 episodes

## Summary

| Metric | State Mode | Video Mode |
|--------|------------|------------|
| MPL (dishes/hr) | 91.2 ± 7.3 | 89.3 ± 7.5 |
| Error Rate (%) | 4.03 ± 2.23 | 5.36 ± 1.60 |
| Wage Parity (ŵᵣ/wₕ) | 1.319 ± 0.188 | 1.224 ± 0.148 |
| Consumer Surplus ($/hr) | 0.02 ± 0.08 | 0.01 ± 0.03 |
| Spread Value ($/hr) | 5.76 ± 3.35 | 4.04 ± 2.65 |

## Interpretation

- **MPL Difference**: -2.0% (video vs state)
- **Error Rate Difference**: +33.2% (video vs state)
- **Wage Parity Difference**: -7.2% (video vs state)

### Sanity Checks

- State mode convergence: ✅ Yes (MPL = 91.2)
- Video mode convergence: ✅ Yes (MPL = 89.3)
- State mode consumer surplus non-negative: ✅ Yes
- Video mode consumer surplus non-negative: ✅ Yes
- Performance similarity (within 20%): ✅ Yes

### Conclusion

✅ **Both modes converge to similar performance.** Economics layer successfully operates on both state and video observations.

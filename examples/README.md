# Examples

Run the standalone demo from the repository root:

```bash
python examples/demo.py
```

The corrected demo now focuses on the additive boundary-based construction used
by the paper instead of the previous signed-distance / normalized-sum fallback.

## What the demo shows

1. **Convex polygon**
   - standard implicit spline contour for a convex control polygon.

2. **Polygon with hole**
   - composition of an outer polygon basis with the complement of a hole basis.

3. **Section 7 paper-style non-convex examples**
   - two non-convex control polygons listed in true CCW boundary order;
   - dashed control polygon and vertex markers overlaid on the contour;
   - contour drawn from the additive polygon basis `imp_spline_2d`;
   - decomposition diagnostics computed from `triangulate_polygon(...)` and
     `convex_decomp_field(...)`, where internal decomposition edges cancel.

4. **Section 9 partition net**
   - a conforming irregular convex-cell partition;
   - programmatic validation via `validate_partition(...)`;
   - additive basis identity `sum_k B_k = B_outer`, without post-hoc
     normalization.

## Output figures

Running the script writes three non-interactive PNG figures into `examples/`:

- `demo_basic_examples.png`
- `demo_section7_paper_style.png`
- `demo_section9_partition.png`

## Notes

- The main contour shown for the spline basis is the `0.5` iso-contour.
- The corrected decomposition path is **additive**. It does not use the old
  smooth-union workaround or the signed-distance fallback from PR #9.
- Partition cells are required to share identical edge endpoints with opposite
  orientation on shared edges; `validate_partition(...)` checks that topology.

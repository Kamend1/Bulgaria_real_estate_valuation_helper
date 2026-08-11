# Track G — manual structured-extraction feasibility pilot (D1+D2 proxy)

**Method:** read 25 real `description_clean` samples directly (no LLM API
— I did the extraction myself as a stand-in), attempting to tag each
against the originally-planned attribute schema (`has_renovation`,
`mentions_view`, `mentions_metro_proximity`, `construction_quality_mention`,
`mentions_parking`). Cost: free, no new infrastructure. Goal: find out if
the schema itself is right before spending anything on an actual LLM
pipeline.

## Headline: feasibility is high, but the original schema was wrong in three ways

**1. Renovation needs 3 states, not a boolean.** Found an explicit
"Апартаментът е за ремонт" (needs renovation) listing — the opposite
polarity from what a boolean `has_renovation` flag assumes. Needs to be
`renovation_status: renovated | needs_renovation | not_mentioned`.

**2. Renovation is often component-specific, not generic.** Real listings
say "С ЧИСТО НОВ ПОКРИВ" (brand new roof), "НОВА ЕЛ. ИНСТАЛАЦИЯ" (new
electrical), "ПВЦ ДОГРАМА ЧАСТИЧНО" (PVC windows, partially) — not just
"ремонт." A single coarse flag loses real signal here. Worth either
sub-attributes (`roof_renovated`, `windows_renovated`, ...) or explicitly
accepting the information loss of one coarse flag for v1.

**3. Metro proximity is often quantified, not just mentioned.** "на 10
минути пеша от метростанция Надежда" — a walking-time number, not just a
yes/no. Worth extracting as `metro_walk_minutes` (numeric) instead of a
boolean when stated.

## Attribute discovered but not in the original schema

- **`is_furnished`** — "изцяло обзаведено" (fully furnished) shows up
  repeatedly and is clearly a real price driver, distinct from renovation
  status. Missing from the original D1 draft; should be added.

## Property-type dependence is bigger than expected

Land plots (`парцел`) have a **completely different relevant attribute
set** — utility/road access, plot shape, ПУП (zoning) status — vs.
residential/commercial buildings (view, renovation, furnished, parking).
A flat schema across all types wastes extraction effort. This maps
naturally onto the AVM's existing 5-segment split — land is already
excluded from the AVM entirely (different pricing unit, per Round 1's
original segmentation call), so this isn't a new problem, just confirms
the segment boundary was drawn in the right place.

## Edge case worth flagging

One listing (`id=148741`) was an **обезщетение (compensation) deal** —
developer gives apartments in exchange for land, not a normal cash sale.
This kind of structural edge case needs explicit handling (most likely:
exclude from extraction/training, same pattern as other bad-data
exclusions already in the pipeline) rather than naive extraction.

## Incidental finding, not part of D1/D2 but worth noting

My manual sample query didn't filter `deal_type_normalized = 'sale'` (a
quick ad-hoc pull, not the production query) and picked up at least one
rental listing. Not a pipeline bug — the actual training queries already
filter this — just a reminder to keep that filter explicit in any future
extraction script, since it's easy to forget in a one-off script.

## Revised D1 schema (supersedes the Round 3 plan's draft)

```
renovation_status:     enum [renovated, needs_renovation, not_mentioned]
renovation_components: list  [roof, electrical, windows, kitchen, bathroom, ...]  (optional, v2)
is_furnished:           bool
mentions_view:          bool
view_type:              enum [sea, park, panorama, courtyard, not_mentioned]  (optional refinement)
metro_walk_minutes:     int | null   (numeric when stated, else null)
construction_quality:   enum [luxury, standard, needs_work, not_mentioned]
mentions_parking:       bool
```

## Conclusion

Worth pursuing as a real Track D pilot **if** an LLM API becomes
available in this environment — the schema is now grounded in what
listings actually say, not guessed. Still need the API-access decision
before D2 (real pilot validation at scale) can move beyond this manual
proxy.

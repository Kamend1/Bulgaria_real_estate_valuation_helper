# Cadastral / zoning open-data pipeline — technical plan

**Status: merged into the main app (2026-08-17).** What started as a
standalone PoC under `gis_cadastre_poc/` is now `utils/gis/` — a regular
part of this codebase, on the same `requirements.txt` as everything else,
with a live panel on the comparables report page and a `scripts/`
CLI entrypoint like every other admin/dev script in this project. See
"What's live in the app" below for the current shipped surface; the rest
of this document is the original protocol research, most of which is
still accurate and is what the current code is built on.

**Headline finding, updated 2026-08-17 — the zoning gap is closed for
Sofia.** Earlier drafts of this document said flatly that no municipal
zoning-parameters API existed anywhere in this project's research. That
was wrong for Sofia specifically: **`isofmap.bg` (GIS-Sofia's own free
public viewer) has a genuine, unauthenticated, working WMS endpoint that
returns real ОУП zoning parameters — Устройствена зона, Плътност на
застрояване, КИНТ, Мин. озеленена площ — per point.** Found by walking
the site's own client-side JS (no docs exist) rather than guessing
endpoint names; see "isofmap.bg zoning connector" below for the full
trace and `utils/gis/connectors/isofmap_client.py` for the working code.
It only covers Sofia (like NAG Sofia's case registry), and only the
"urbanized territory" extent of the 2009 ОУП, but within that scope it's
real, live, and now in production.

The cadastral side (АГКК/АGKK — parcel + building geometry) remains a
genuinely free, open, unauthenticated, well-structured national API. The
*other* 264 municipalities' zoning data remains exactly as uncertain as
before — no common schema, no common catalog, unverified reachability.
Any plan that treats "Sofia works" as "therefore the rest of Bulgaria
works the same way" will
over-promise on the zoning half. Everything below reflects that asymmetry
honestly rather than glossing over it.

---

## Step 1 — Technical & protocol assessment

### 1.1 AGKK / INSPIRE cadastral data — VERIFIED LIVE, free, no auth

Live-tested during this research (2026-08, against
`inspire.cadastre.bg`), not assumed from documentation:

| | |
|---|---|
| Cadastral parcels | `GET https://inspire.cadastre.bg/arcgis/rest/services/Cadastral_Parcel/MapServer/0/query` |
| Buildings | `GET https://inspire.cadastre.bg/arcgis/rest/services/Building/MapServer/0/query` |
| Protocol | Esri ArcGIS Server REST (`f=geojson` or `f=json`), **not** a bare OGC WFS endpoint, though a `WFSServer` extension is also advertised (see 1.1.3) |
| Auth | None. Confirmed: a plain `GET` with `where=id_localid='15285.14.122'` returned real geometry + attributes with no key, header, or session |
| Rate limiting | Not documented publicly; `maxRecordCount: 1000` per query is enforced server-side (`exceededTransferLimit: true` flag when hit — paginate with `resultOffset`) |
| CRS | Native: **EPSG:4258** (ETRS89 geographic). Confirm/transform explicitly for anything downstream expecting EPSG:4326, EPSG:32635 (UTM 35N), or EPSG:7801 (BGS2005, Bulgaria's own official CRS) — see `spatial_engine/geometry_ops.py` |
| Key fields (Cadastral_Parcel) | `id_localid` (the ПИ identifier, e.g. `"15285.14.122"` — exact EKATTE.masiv.imot format), `areavalue` (m²), `admunit`, `nationalcadastralref`, `zoning` (present but not verified to carry detailed regulation data — likely a coarse INSPIRE CadastralZoning reference, not Kint/density) |

#### 1.1.1 Sample request/response (as actually observed)

```
GET https://inspire.cadastre.bg/arcgis/rest/services/Cadastral_Parcel/MapServer/0/query
    ?where=id_localid='15285.14.122'
    &outFields=*
    &f=geojson
    &returnGeometry=true
```

Response: GeoJSON `FeatureCollection`, one polygon feature, 25 attribute
fields including `areavalue: 1471.0`, `label: "122"`, coordinates in
EPSG:4258 decimal degrees.

#### 1.1.2 Do not confuse with AGKK's *paid* WMS product

The КАИС portal separately advertises service **#8002** — "Достъп до
кадастрални данни чрез WMS услуга" — which is a **different, paid**
product:

- Requires an application through the КАИС e-services portal, signed with
  a qualified electronic signature (КЕП)
- Requires declaring the access IP address in advance (whitelisting)
- Up to 3 business days for approval
- **€40.90/month or €409.03/year, per layer, per object type**

This PoC does not use that service and does not need to — the free
INSPIRE download service above already provides parcel/building geometry
and core attributes with no application process. Reserve the paid product
for scenarios needing something the free service doesn't have (e.g. an
official signed extract for legal purposes, not raw geometry for
analysis).

#### 1.1.3 OGC WFS path — advertised, not fully verified

The service's own capabilities document lists `WFSServer` under
`supportedExtensions`, meaning a standards-compliant
`GetCapabilities`/`GetFeature` path likely exists at
`.../Cadastral_Parcel/MapServer/WFSServer`. A `GetCapabilities` request
against it returned HTTP 400 during this research pass (likely a
parameter-casing or version-negotiation quirk specific to Esri's WFS
implementation, not a sign the service is unavailable — the same
MapServer answered the REST query above without issue). `OWSLib` is
included in `requirements-gis.txt` as an alternate path for teams that
specifically want OGC-standard access (e.g. to reuse the same client
against a future GeoServer-based source), but **this PoC's default and
tested path is the plain ArcGIS REST `query` operation**, which is
simpler, confirmed working, and returns GeoJSON directly.

### 1.2 data.egov.bg (CKAN) — verified as a real CKAN deployment

- Base: `https://data.egov.bg`, confirmed CKAN (standard facets:
  organizations/tags/formats/licenses; a published "API спецификация"
  page linked from the dataset browser)
- Standard CKAN Action API applies: `GET /api/3/action/package_search?q=...`,
  `GET /api/3/action/package_show?id=...` — no key needed for these
  read-only actions
- **What it's actually useful for here:** a *catalog*, not a live zoning
  API. Use it to discover which municipalities have published an ОУП/
  zoning dataset and pull the resource URL (often a Shapefile/CSV/GeoJSON
  download, occasionally a link to a live WFS/ArcGIS service) — then feed
  any service URL it turns up into the municipal zoning connector. Do not
  expect `package_search` itself to answer "what's the Kint at this
  point" — it indexes datasets, not features.
- 11,684+ datasets on the portal at time of research; CSV is by far the
  dominant format (~68%), meaning most "open data" from municipalities
  is static tabular exports refreshed on some unpublished cadence, not a
  live queryable service — plan the fallback path (periodic batch
  ingestion of a CSV/SHP) alongside the live-query path, not as an
  afterthought.

### 1.3 Municipal zoning (Sofia / Plovdiv / Varna) — the honest gap

This is the part of the brief where "free open-data API" does not
actually hold nationally, and it's worth being direct about it rather
than papering over it with a plausible-looking mock:

- **Sofia**: tried across two separate research passes (2026-08-11 and
  2026-08-17), including endpoint URLs supplied directly by the user on
  the second pass (`nag.sofia.bg/arcgis/rest/services`,
  `nag.sofia.bg/arcgis/rest/services/OUP/MapServer`, `gis.sofia.bg`).
  **None resolved live**: `gis.sofiaplan.bg/arcgis/rest/services` 404s,
  `nag.sofia.bg/arcgis/rest/*` serves the plain HTML site (not an ArcGIS
  Server at that path), and `gis.sofia.bg` has no DNS record at all.
  Traced further to `nag.sofia.bg/OpenMap/Zones`, which is almost
  certainly the real live viewer (Sofia's "Карта на административните
  актове") — but it's a JavaScript single-page app whose actual API
  calls happen client-side and are invisible to static HTTP fetching.
  **The concrete unblock here is a real browser**: open that map, open
  DevTools' Network tab, click a zone, and read the resulting request
  URL — something no amount of further guessing at hostnames will
  substitute for. Still templated in `config/municipalities.py` as
  `verified_live=False`.
- **Plovdiv, Varna**: no public WFS/ArcGIS REST endpoint was located
  during this research pass. The realistic next step for either is (a)
  check data.egov.bg for that municipality's own published zoning
  dataset via the CKAN connector, or (b) contact the municipality's own
  GIS/ГИС department directly — many Bulgarian municipalities' spatial
  planning data is published only through their own portal's web viewer,
  with no documented machine-readable API behind it.
- **Architectural consequence:** `municipal_zoning_client.py` is written
  generically (point-in-polygon query against any Esri
  MapServer/FeatureServer, given a service URL + layer ID + field
  mapping) specifically so that onboarding a new municipality is a config
  change, not a code change — but every entry needs its endpoint
  independently confirmed and its field schema independently inspected
  (via `inspect_layer_schema()`) before it can return trustworthy
  numbers. There is no way to ship a "just works for all of Bulgaria"
  zoning connector today because the underlying data landscape doesn't
  have that shape yet.

### 1.4 CRS transformations needed

| From | To | When |
|---|---|---|
| EPSG:4258 (AGKK native) | EPSG:4326 (WGS84) | Feeding coordinates to consumer-facing maps/APIs that assume WGS84 (the practical offset is sub-meter, but a correct pipeline transforms explicitly rather than assuming equivalence) |
| EPSG:4258 | EPSG:32635 (UTM 35N) | Any client-side metric area/distance calculation |
| EPSG:4326 (typical GPS input) | EPSG:4258 or municipal-native CRS | Before querying AGKK/municipal services by point, since their spatial filters expect their own native CRS in `inSR` |
| Municipal-native (often EPSG:7801, BGS2005) | EPSG:4326 | Displaying municipal zoning geometry on a standard web map |

All handled centrally in `spatial_engine/geometry_ops.py` via cached
`pyproj.Transformer` instances — never ad hoc per connector.

### 1.5 Fallback strategy when an endpoint is down

- **AGKK**: retry is not currently implemented beyond a single attempt
  (`AgkkClientError` propagates immediately) — appropriate for a PoC used
  interactively; a production version should add bounded retry with
  backoff for transient 5xx/timeout, but should still surface a hard
  failure rather than silently returning stale/cached data as if it were
  fresh (the cache is opt-in and explicit about `status: "cached"`, never
  silently substituted on a live-request failure)
- **Municipal zoning**: expected to fail more often, given §1.3 — the
  orchestrator (`poc_lookup_parcel.py`) treats a zoning failure as
  non-fatal: it still returns the parcel geometry half of the report with
  `zoning.confidence: "unavailable"` and an error detail in `sources[]`,
  rather than aborting the whole report
- **data.egov.bg**: same non-fatal treatment appropriate if wired into
  the orchestrator (currently a standalone connector, not yet called from
  `poc_lookup_parcel.py` — see "Not yet wired up" below)

---

## Step 2 — Module architecture (as shipped)

```
utils/gis/
  connectors/
    agkk_client.py            # AGKK INSPIRE Cadastral_Parcel / Building — VERIFIED LIVE
                               #   fetch_parcel_by_cadastral_id / _by_coordinates
                               #   fetch_neighbouring_parcels (esriSpatialRelIntersects,
                               #     see note below — Touches returns nothing on real data)
                               #   fetch_building_by_cadastral_id / fetch_buildings_on_parcel
                               #     (id_localid prefix match — buildings' ids are literally
                               #     "<parent parcel id>.<n>", confirmed live)
    ckan_client.py             # data.egov.bg CKAN Action API — dataset catalog search
    municipal_zoning_client.py # generic Esri REST point-in-polygon zoning query,
                               #   + discover_services()/inspect_layer_schema() for
                               #   onboarding a new municipality without guessing
  spatial_engine/
    geometry_ops.py           # pyproj CRS transforms; shapely point-in-polygon
                               #   disambiguation; compute_metric_area_perimeter
                               #   (Building has no areavalue field — this derives it);
                               #   esri_rings_from_geojson (GeoJSON -> Esri query geometry)
  models/
    schemas.py                 # CadastralIdentifier, ParcelGeometry, BuildingInfo,
                               #   NeighbourParcel, ZoningInfo, LegalDescription, SourceMeta
  cache/
    sqlite_cache.py           # GisCache + get_shared_cache() — process-wide instance
                               #   used from FastAPI request handlers
  config/
    municipalities.py         # MunicipalZoningConfig registry — the pluggable part
  engines/
    parcel_engine.py          # get_parcel_with_buildings, get_parcel_with_neighbours
    building_engine.py        # get_building_profile, total_built_up_area_sqm
    legal_description_engine.py  # generate_legal_description — the "при граници: ..."
                               #   legal text generator, built on parcel_engine's neighbours

app/services/gis_service.py   # FastAPI-facing adapter: AppraisalReport -> engines,
                               #   no DB access needed (all data is live from AGKK)
app/routers/comparables.py    # wires gis_service into GET /comparables/ and
                               #   POST /comparables/save-legal-description
app/templates/comparables/_cadastre_panel.html  # the UI panel
scripts/lookup_parcel.py      # CLI entrypoint (python -m scripts.lookup_parcel), matches
                               #   the convention of every other scripts/*.py in this repo
```

Why each module is separated the way it is:

- **connectors/** hold nothing but HTTP + response parsing — no spatial
  math, no business rules. Each raises its own `*Error` type rather than
  returning `None`/empty on failure, so a caller can't accidentally treat
  "service down" as "no data here."
- **spatial_engine/** is the only place shapely/pyproj get imported for
  actual geometry work, so CRS bugs have one place to be fixed, not five.
- **models/** is the contract between connectors and engines —
  `ZoningInfo.confidence` exists specifically so "we found a matching
  zone" and "we don't have zoning data for this location" are never
  conflated into the same null-filled shape.
- **cache/** wraps every connector's raw response, not just AGKK's,
  because data.egov.bg and municipal servers are equally shared public
  infrastructure that shouldn't be hammered during iteration.
- **config/** isolates the one part of the system (municipal endpoints)
  that has no stable, verifiable-once answer.
- **engines/** is new since the merge: it's where cross-connector
  orchestration and appraisal-specific logic live (e.g. "sum every
  building's area on a parcel," "turn a parcel + its neighbours into
  legal Bulgarian text") — kept separate from `connectors/` so the raw
  HTTP clients stay dumb and reusable.

### The neighbours mechanism (Track 2 of the "make it coherent" ask)

`fetch_neighbouring_parcels()` queries AGKK with the subject parcel's own
polygon and `spatialRel=esriSpatialRelIntersects`, filtering the subject
parcel itself out of the result client-side. **Not**
`esriSpatialRelTouches`, despite that reading as the "more correct" OGC
predicate for "shares a boundary" — tested live and confirmed `Touches`
returns zero results even against parcels visibly adjacent in the data,
because real surveyed cadastral boundaries essentially never share exact
vertex-for-vertex topology. `Intersects` correctly returns the subject
parcel plus every true neighbour.

### Buildings have no area field — computed, not fetched

Confirmed live: `BU.Building` has no `areavalue` attribute (unlike
`CP.CadastralParcel`, which does). `agkk_client._feature_to_building()`
derives it via `spatial_engine.compute_metric_area_perimeter()` —
reproject the polygon into EPSG:32635, then read `.area`. This is also
why `shapely`/`pyproj` became real (not optional) dependencies of the
main app rather than staying PoC-only extras.

### Not yet wired up

`ckan_client.py` is implemented and usable standalone but not called from
the FastAPI app — its job (discovering datasets/endpoints) is an
investigative step a developer runs once per municipality while
populating `config/municipalities.py`, not a per-request runtime step.

---

## What's live in the app

- **Comparables report page** (`/comparables/`): a "Кадастрален
  идентификатор" field on the subject form, and — once filled in — a
  "Кадастър и правно описание на имота" panel showing the parcel's AGKK
  area, every building AGKK has registered on it (floors, dwellings,
  computed footprint area), a generated legal description with the
  boundary-neighbour clause (building-led when a specific building id was
  entered), a schematic SVG sketch of the real parcel boundary, **live ОУП
  zoning parameters for Sofia parcels** (Устройствена зона / Плътност /
  КИНТ / Мин. озеленена площ, via isofmap.bg — see below), and any
  related НАГ development-plan cases for that parcel. Legal description
  is editable before saving, same pattern as the AVM panel's "accept or
  override."
- **`scripts/lookup_parcel.py`**: CLI for ad-hoc lookups outside the web
  UI — `python -m scripts.lookup_parcel --cadastral-id ...`.
- Verified end-to-end through the running app (not just the engine layer
  in isolation) across every feature above, including the zoning panel
  showing real values (Смф, 60% density, Kint 3.5, 40% landscaping) for a
  real Sofia parcel, and an invalid cadastral id degrading to a clean
  "not found" state rather than a 500.
- **Not done**: the legal description isn't yet pulled into the DOCX/
  Excel export templates — it's captured on `AppraisalReport.legal_description`
  but the export code doesn't render it yet.

---

## isofmap.bg zoning connector — the ОУП data source (2026-08-17)

**This closes the single biggest open question of the whole project**:
whether real Kint/density/height zoning parameters are reachable at all
for any Bulgarian municipality. For Sofia, yes.

Found because the user described their own manual workflow on
isofmap.bg (no login, search a parcel, tick the "Устройство на
територията" layer checkbox — off by default — then click the "i"
button) precisely enough to point the investigation at the right site.
Static fetching of the map page itself showed nothing (same JS-SPA wall
hit earlier with `nag.sofia.bg/OpenMap/Zones`), so this was cracked by
reading the site's own client-side source instead:

1. `js/mapUrl.js` → `mapUrl()` returns `isofmap.bg/owsmap` — a WMS
   endpoint (confirmed via `GetCapabilities`: UMN **MapServer 8.0**, not
   Esri/ArcGIS).
2. `js/olCustomControls.js` → `Info.prototype.getFeatureInfo` — the
   map's click-to-identify handler does nothing custom, just standard
   OGC `GetFeatureInfo` (`INFO_FORMAT=text/html`) per visible+queryable
   layer, built via OpenLayers' own `getSource().getFeatureInfoUrl()`.
   `js/layersDefinition.js` line ~162 confirms `selectable: layer.queryable`
   — i.e. the "is this layer clickable" flag is taken directly from the
   WMS capabilities' own `queryable` attribute, which is why a layer
   named literally `zoning` (queryable="0") is a dead end — it's a
   label/cartography-only layer, and the *site's own official UI* never
   queries it either.
3. The real data layer was found by walking the `<Layer>` tree in
   `GetCapabilities` for anything titled "Устройство на територията" or
   "Общ устройствен план" (their `<Name>` values are themselves Cyrillic
   strings — the reason earlier ASCII-only guesses like `zoning` and
   `z_structurezone` never found it). The zoning-group layer itself isn't
   queryable (`msPostGISLayerOpen(): Nothing specified in DATA statement`
   — it's a pure UI container); its actual leaf child is
   **`gdp_close_2010`** ("Урбанизирани територии ... (ОУП)"), which *is*
   queryable and returns the full parameter table.
4. Two MapServer-specific quirks, found only by iterating on
   `ServiceException` responses (undocumented anywhere): `STYLES=`
   (even empty) is a required GetFeatureInfo parameter for this instance;
   and the HTML response is genuine UTF-8 bytes with no charset in the
   `Content-Type` header, so `requests`' auto-detection (falls back to
   Latin-1) silently mangles it — must decode `response.content` as
   UTF-8 explicitly (a milder version of the imot.bg Windows-1251 gotcha
   already documented in this repo's `CLAUDE.md`).

**Verified as an exact match**, not just "plausible output": queried a
real Sofia parcel (68134.905.1462, Лозенец) and got back zone code
"Смф", density 60%, Kint 3.5, min. landscaping 40%, and the *exact same
purpose/description sentence, word for word*, as a table the user
obtained independently by hand on the live site.

Implemented in `utils/gis/connectors/isofmap_client.py`. Coverage is
real but bounded: `gdp_close_2010` only has data within the 2009 ОУП's
mapped "urbanized territory" extent — a point outside that (or on a
green-belt/landscaping sub-zone with genuinely 0% density and no Kint)
correctly returns an empty result, which is the right answer for that
location, not a bug.

### Not investigated further

- `proposal_pup` (under "Устройство на територията (ПУП)") — draft/
  proposed ПУП changes, queryable, returned empty at every point tried
  (plausible — draft proposals are sparse) but not confirmed with a real
  hit.
- `reg_border` — returned real "Местност по регулация" + governing
  order/decision data (a different, complementary field set to the
  zoning table) but isn't wired into the connector; would be a
  reasonable follow-up if that data proves useful.
- `gdp_specificrulesareas`, `gdp_limitvalues`, `gdp_far_2010` — sibling
  layers under the same ОУП group, plausibly override/exception zones
  for specific areas, not tested for a positive hit.
- The generic Esri-REST-based `municipal_zoning_client.py` +
  `config/municipalities.py` machinery built earlier is now superseded
  *for Sofia* by this dedicated connector — left in place unmodified as
  the right shape for a future municipality that *does* run ArcGIS REST
  (Plovdiv/Varna's actual stack is still unknown).

---

## Legal-text formatting + parcel sketch (2026-08-17)

Three follow-ups, informed by the user's own separate chsi-app project
(github.com/Kamend1/chsi-app — a ЗКИР-standard legal description
generator for Bulgarian judicial executors), whose conventions we now
match rather than reinvent:

1. **Digit-spelled identifiers + word-spelled areas.** Every cadastral
   number and area figure in the generated text now appears as
   `{numeral} /{spelled out in words}/` — e.g. `68134.905.1462 /шест,
   осем, едно, ... /` and `1625 кв.м /хиляда шестстотин двадесет и пет
   квадратни метра/` — matching chsi-app's `describe_pi()` convention
   (standard Bulgarian notarial practice for identification numbers).
   Ported chsi-app's `numwords.py` into `utils/gis/engines/numwords_bg.py`
   rather than re-deriving Bulgarian number grammar from scratch.
2. **Building-led description.** When the entered id is a 4-segment
   building id, the primary generated text now switches to
   `legal_description_engine.generate_building_description()` — "СГРАДА
   с идентификатор ... която сграда е разположена в поземлен имот с
   идентификатор ..." — matching chsi-app's `describe_building()`
   template, instead of always describing the bare parcel regardless of
   what was actually selected.
3. **Parcel shape rendering.** `utils/gis/spatial_engine/sketch_svg.py`
   renders the parcel's real AGKK boundary polygon as an inline SVG
   (grid background, translucent fill, centered id + area labels) —
   adapted from chsi-app's `sketch.py`, but using genuine fetched
   geometry instead of that project's sample/placeholder points. Does
   NOT attempt to label neighbours along specific edges the way
   chsi-app's prototype does — we only know *which* parcels are
   neighbours, not which edge of the subject polygon each one borders,
   and a wrong guess there would be worse than no label.

Not ported from chsi-app: the КККР approval-order clause
(`_kkkr_order()`) and full notarial address composition (`_address()`) —
neither is available from AGKK's free INSPIRE service, so adding them
would mean fabricating placeholder text rather than real data.

## NAG Sofia development-plan case registry (2026-08-17)

Not a zoning-parameters (Kint/density/height) source — see §1.3, that
still doesn't exist for any municipality. What it is: a real, live,
working connector to Sofia's НАГ case registry of ПУП applications and
administrative decisions, filterable by cadastral identifier, with
genuine downloadable case documents (Скица – предложение, Заявление,
ИПР, ИПРЗ, РУП, ...) — confirmed by downloading and validating an actual
PDF (`%PDF-1.7` header, correct `Content-Type`).

Found by testing a URL the user had gotten from elsewhere
(`POST /SearchDevelopmentPlans/Search` with a JSON `CadastralIdentifier`
body) — which doesn't actually work: it returns `200` with real page
content, but silently ignores the filter and shows the 10 most recent
citywide cases regardless of input. Confirmed this by testing a known-
real cadastral id and a fake one and seeing identical row counts. The
real mechanism, reverse-engineered from the live search page's rendered
form config (no public API docs exist for this):

- **GET**, not POST — the form's `data-ajax-method` is `GET` despite
  `<form method="post">`
- Filter field is **`Identifier`**, not `CadastralIdentifier`
- Requires a session-bound `searchQueryId` GUID, minted per page load —
  `GET /SearchDevelopmentPlans` first (establishes the session +
  extracts the id from that response), then
  `GET /SearchDevelopmentPlans/Search?searchQueryId=<id>&Identifier=<cadastral id>`
  with the same cookies
- Document links: each result row's raw "Attachments" cell is a
  `;`-separated `name&path` list; real download URL is
  `/SearchDevelopmentPlans/ViewAttachment?filename=<path>`

Implemented in `utils/gis/connectors/nag_sofia_client.py` (parses the
Kendo grid's HTML via BeautifulSoup — already a project dependency, no
new package needed) and surfaced as "Свързани устройствени планове" on
the cadastre panel. Cached like every other connector — 2 GETs per
uncached lookup, called unconditionally (no EKATTE prefix filter,
since Sofia spans more than one EKATTE-style code and a wrong filter
would silently hide real Sofia parcels — confirmed empty results for
non-Sofia parcels are harmless, just an extra round-trip).

## Open items

1. **Wire the legal description into DOCX/Excel export.**
2. **Decide the CSV/batch fallback** for municipalities that only publish
   static zoning exports via data.egov.bg rather than a live service.
3. **Pagination** for AGKK queries that exceed `maxRecordCount: 1000`.
4. **Retry/backoff** on transient network failures — currently
   single-attempt by design.
5. **Plovdiv/Varna zoning** — still completely unverified; isofmap.bg's
   discovery method (read the site's own client JS for the real WMS/API
   calls) is the template to try if either municipality has a similarly
   free public viewer.
6. Follow up on the untested isofmap.bg sibling layers noted above
   (`reg_border`, `gdp_specificrulesareas`, etc.) if richer per-parcel
   detail is ever needed beyond the current Kint/density/landscaping set.

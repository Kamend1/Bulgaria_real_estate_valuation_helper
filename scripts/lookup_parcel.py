"""
CLI for the cadastral/zoning/legal-description pipeline (utils/gis).
Mainly useful for ad-hoc lookups and debugging outside the web UI — the
web UI (comparables report's "Кадастър" panel) is the primary interface.

Usage:
    python -m scripts.lookup_parcel --cadastral-id 15285.14.122
    python -m scripts.lookup_parcel --cadastral-id 15285.14.122 --settlement-name "гр. София"
    python -m scripts.lookup_parcel --lat 42.6977 --lon 23.3219
    python -m scripts.lookup_parcel --building-id 15285.13.286.2
"""
import argparse
import sys

sys.path.insert(0, ".")

from utils.gis.cache.sqlite_cache import GisCache
from utils.gis.connectors.agkk_client import AgkkClientError
from utils.gis.engines import building_engine, legal_description_engine, parcel_engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cadastral-id", help="parcel id, e.g. 15285.14.122")
    parser.add_argument("--building-id", help="building id, e.g. 15285.13.286.2 (looked up on its own)")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--settlement-name", help="free text for the legal description, e.g. 'гр. София'")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cache = None if args.no_cache else GisCache()

    try:
        if args.building_id:
            building = building_engine.get_building_profile(args.building_id, cache=cache)
            print(building.model_dump_json(indent=2))
            return 0

        if args.cadastral_id:
            parcel, buildings = parcel_engine.get_parcel_with_buildings(cadastral_id=args.cadastral_id, cache=cache)
        elif args.lat is not None and args.lon is not None:
            parcel, buildings = parcel_engine.get_parcel_with_buildings(lat=args.lat, lon=args.lon, cache=cache)
        else:
            parser.error("Provide --cadastral-id, --lat/--lon, or --building-id")
            return 1

        print(parcel.model_dump_json(indent=2))
        print(f"\n{len(buildings)} building(s) on this parcel:")
        for b in buildings:
            print(f"  {b.cadastral_id}  floors={b.floors_above_ground}  area~{b.area_sqm:.1f} m^2" if b.area_sqm else f"  {b.cadastral_id}")

        legal = legal_description_engine.generate_legal_description(
            parcel.cadastral_id, settlement_name=args.settlement_name, cache=cache
        )
        print("\nLegal description:")
        print(legal.text_bg)
        return 0
    except (AgkkClientError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    sys.exit(main())

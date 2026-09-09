"""Thin CLI: per-coupon fusion driver (stub)."""

from weldfusion.pipeline.run_coupon import run_coupon

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("coupon_id")
    args = parser.parse_args()
    run_coupon(args.coupon_id)

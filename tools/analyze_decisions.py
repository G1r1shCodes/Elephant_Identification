import os
import sys
import pandas as pd
import numpy as np


def analyze(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'")
        print(
            "Please ensure you have run the UI and made some merge/reject decisions first."
        )
        return

    print(f"\n--- Analyzing {csv_path} ---\n")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("Log file is empty. Run the UI and make decisions first.")
        return

    merged = df[df["decision"] == "merged"]
    rejected = df[df["decision"] == "rejected"]

    print(
        f"Total decisions logged: {len(df)} (Merged: {len(merged)}, Rejected: {len(rejected)})\n"
    )

    # 1. Basic distributions
    print("=== 1. DISTRIBUTIONS (Effective Score) ===")
    if not merged.empty:
        print(
            f"MERGED:   mean={merged['effective_score'].mean():.4f} | std={merged['effective_score'].std():.4f} | min={merged['effective_score'].min():.4f} | max={merged['effective_score'].max():.4f}"
        )
    else:
        print("MERGED:   No data yet.")

    if not rejected.empty:
        print(
            f"REJECTED: mean={rejected['effective_score'].mean():.4f} | std={rejected['effective_score'].std():.4f} | min={rejected['effective_score'].min():.4f} | max={rejected['effective_score'].max():.4f}"
        )
    else:
        print("REJECTED: No data yet.")
    print()

    # 2. Percentile thresholds
    print("=== 2. CALIBRATED THRESHOLDS ===")
    safe_threshold = None
    if not merged.empty:
        safe_threshold = np.percentile(merged["effective_score"], 25)
        print(f"Suggested SAFE threshold (25th pct merged):   {safe_threshold:.4f}")
    if not rejected.empty:
        reject_threshold = np.percentile(rejected["effective_score"], 75)
        print(f"Suggested REJECT threshold (75th pct reject): {reject_threshold:.4f}")
    print()

    # 3. Overlap check
    print("=== 3. OVERLAP CHECK ===")
    if not merged.empty and not rejected.empty:
        overlap_low = max(
            rejected["effective_score"].min(), merged["effective_score"].min()
        )
        overlap_high = min(
            rejected["effective_score"].max(), merged["effective_score"].max()
        )

        if overlap_low < overlap_high:
            print(f"⚠ OVERLAP DETECTED: from {overlap_low:.4f} to {overlap_high:.4f}")
            print("  → Model is not perfectly separated. Rely on thresholds carefully.")
        else:
            print("✔ No overlap detected. Classes are well separated.")
    else:
        print("Not enough data to check overlap.")
    print()

    # 4. Precision of "safe zone"
    print("=== 4. SAFE ZONE PRECISION ===")
    if safe_threshold is not None:
        safe_preds = df[df["effective_score"] >= safe_threshold]
        if len(safe_preds) > 0:
            precision = (safe_preds["decision"] == "merged").mean()
            print(
                f"Precision for score >= {safe_threshold:.4f}: {precision:.1%} ({len(safe_preds[safe_preds['decision'] == 'merged'])}/{len(safe_preds)})"
            )
            if precision >= 0.90:
                print("  → ✔ Safe to consider auto-merge for this tier.")
            elif precision >= 0.75:
                print("  → ⚠ Use 'Recommended merge' (requires user confirmation).")
            else:
                print("  → ✖ Do NOT automate. Fix scoring first.")
        else:
            print("No decisions fall into the safe zone.")
    else:
        print("Cannot compute precision without merged data.")
    print()

    # 5. Feature importance
    print("=== 5. FEATURE CORRELATION (Sanity Check) ===")
    if not merged.empty and not rejected.empty:
        df_corr = df.copy()
        df_corr["decision_binary"] = (df_corr["decision"] == "merged").astype(int)

        if "was_singleton" in df_corr.columns:
            # Handle string 'True'/'False' vs boolean True/False
            if df_corr["was_singleton"].dtype == object:
                df_corr["was_singleton"] = df_corr["was_singleton"].map(
                    {"True": 1, "False": 0, "true": 1, "false": 0, True: 1, False: 0}
                )
            else:
                df_corr["was_singleton"] = df_corr["was_singleton"].astype(int)

        numeric_cols = [
            "effective_score",
            "direct_score",
            "bridge_score",
            "cohesion",
            "was_singleton",
            "decision_binary",
        ]
        numeric_cols = [c for c in numeric_cols if c in df_corr.columns]

        corrs = (
            df_corr[numeric_cols]
            .corr()["decision_binary"]
            .drop("decision_binary")
            .sort_values(ascending=False)
        )
        for col, val in corrs.items():
            print(f"{col:>16}: {val:+.4f}")
    else:
        print("Need both merged and rejected examples for correlation.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "merge_decisions.csv"
    analyze(target)

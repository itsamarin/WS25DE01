"""
Output Verification Script
Verifies that all required outputs have been generated
"""

import os
from pathlib import Path

print("="*70)
print("ACADEMIC PERFORMANCE PREDICTION - OUTPUT VERIFICATION")
print("="*70)

# Define expected outputs
expected_tables = [
    "tables/RQ1_Table1.xlsx",
    "tables/RQ3_Table1.xlsx"
]

expected_figures = [
    # RQ1: 4 figures
    "figures/RQ1_Fig1.pdf",
    "figures/RQ1_Fig2.pdf",
    "figures/RQ1_Fig3.pdf",
    "figures/RQ1_Fig4.pdf",
    # RQ2: 5 figures
    "figures/RQ2_Fig1.pdf",
    "figures/RQ2_Fig2.pdf",
    "figures/RQ2_Fig3.pdf",
    "figures/RQ2_Fig4.pdf",
    "figures/RQ2_Fig5.pdf",
    # RQ3: 4 figures
    "figures/RQ3_Fig1.pdf",
    "figures/RQ3_Fig2.pdf",
    "figures/RQ3_Fig3.pdf",
    "figures/RQ3_Fig4.pdf",
    # RQ4: 6 figures
    "figures/RQ4_Fig1.pdf",
    "figures/RQ4_Fig2.pdf",
    "figures/RQ4_Fig3.pdf",
    "figures/RQ4_Fig4.pdf",
    "figures/RQ4_Fig5.pdf",
    "figures/RQ4_Fig6.pdf"
]

# Verify tables
print("\n" + "-"*70)
print("TABLES (2 XLSX files)")
print("-"*70)
tables_ok = True
for table in expected_tables:
    exists = os.path.exists(table)
    status = "✓" if exists else "✗"
    print(f"  {status} {table}")
    if not exists:
        tables_ok = False

# Verify figures
print("\n" + "-"*70)
print("FIGURES (19 PDF files)")
print("-"*70)

# Group by RQ
rq_groups = {
    "RQ1": expected_figures[0:4],
    "RQ2": expected_figures[4:9],
    "RQ3": expected_figures[9:13],
    "RQ4": expected_figures[13:19]
}

figures_ok = True
for rq, figs in rq_groups.items():
    print(f"\n{rq} ({len(figs)} figures):")
    for fig in figs:
        exists = os.path.exists(fig)
        status = "✓" if exists else "✗"
        print(f"  {status} {fig}")
        if not exists:
            figures_ok = False

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

tables_count = sum(1 for t in expected_tables if os.path.exists(t))
figures_count = sum(1 for f in expected_figures if os.path.exists(f))

print(f"\nTables:  {tables_count}/{len(expected_tables)} {'✓ COMPLETE' if tables_ok else '✗ INCOMPLETE'}")
print(f"Figures: {figures_count}/{len(expected_figures)} {'✓ COMPLETE' if figures_ok else '✗ INCOMPLETE'}")

if tables_ok and figures_ok:
    print("\n" + "="*70)
    print("✓ ALL OUTPUTS SUCCESSFULLY GENERATED!")
    print("="*70)
    print("\nProject Requirements Met:")
    print("  ✓ RQ1: 1 table + 4 figures")
    print("  ✓ RQ2: 5 figures")
    print("  ✓ RQ3: 1 table + 4 figures")
    print("  ✓ RQ4: 6 figures")
    print("\nTotal: 2 tables + 19 figures")
else:
    print("\n" + "="*70)
    print("✗ SOME OUTPUTS ARE MISSING")
    print("="*70)
    print("\nPlease run:")
    print("  python3 run_simple_analysis.py")
    print("  python3 generate_all_figures.py")

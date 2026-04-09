#!/usr/bin/env python
# coding: utf-8
"""
Battle-hardened Scopus CSV cleanup script.
Handles malformed metadata, type coercion, garbage values, and formatting issues.
"""

import pandas as pd
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load raw data
df = pd.read_csv("scopus_export_final.csv")
print(f"[1] Loaded: {len(df)} rows")

# ============================================================================
# COLUMN SELECTION & REORDERING
# ============================================================================
required_cols = ['Authors', 'Title', 'Abstract', 'Author Keywords', 'Cited by', 'Source title', 'Year']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"WARNING: Missing columns: {missing_cols}")

df = df[required_cols].copy()
print(f"[2] Selected columns: {list(df.columns)}")

# ============================================================================
# NUMERIC COLUMN CLEANUP
# ============================================================================

# Year: convert, filter for realistic range, drop NaN
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df[df['Year'].notna()]
df = df[(df['Year'] >= 1900) & (df['Year'] <= pd.Timestamp.now().year)]
df['Year'] = df['Year'].astype(int)
print(f"[3] Year cleaned: {len(df)} rows, range {df['Year'].min()}-{df['Year'].max()}")

# Cited by: convert, filter for non-negative, drop NaN
df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce')
df = df[df['Cited by'].notna()]
df = df[df['Cited by'] >= 0]
df['Cited by'] = df['Cited by'].astype(int)
print(f"[4] Cited by cleaned: {len(df)} rows, range {df['Cited by'].min()}-{df['Cited by'].max()}")

# ============================================================================
# TEXT COLUMN CLEANUP
# ============================================================================

# Authors: strip whitespace, remove NaN rows, remove rows with garbage chars
df['Authors'] = df['Authors'].str.strip()
df = df[df['Authors'].notna()]
df = df[df['Authors'].str.len() > 0]
print(f"[5] Authors cleaned: {len(df)} rows")

# Title: strip, remove NaN, ensure reasonable length
df['Title'] = df['Title'].str.strip()
df = df[df['Title'].notna()]
df = df[df['Title'].str.len() > 5]  # Titles should be > 5 chars
print(f"[6] Title cleaned: {len(df)} rows")

# Abstract: strip, keep NaN (abstracts may legitimately be missing)
df['Abstract'] = df['Abstract'].str.strip()
# Don't drop NaN here — abstracts are often missing in Scopus
print(f"[7] Abstract stripped: {len(df)} rows (NaN allowed)")

# Author Keywords: strip, keep NaN (keywords often missing)
df['Author Keywords'] = df['Author Keywords'].str.strip()
print(f"[8] Author Keywords stripped: {len(df)} rows (NaN allowed)")

# Source title: strip, remove NaN, remove garbage (page numbers, etc)
df['Source title'] = df['Source title'].str.strip()
df = df[df['Source title'].notna()]
df = df[df['Source title'].str.len() > 0]
# Remove rows where source is just numbers/page ranges (e.g., "pp. 528-542")
df = df[~df['Source title'].str.match(r'^\s*(?:pp\.|pages?|p\.)?\s*\d+', na=False)]
print(f"[9] Source title cleaned: {len(df)} rows")

# ============================================================================
# OUTLIER & DUPLICATE DETECTION
# ============================================================================

# Remove rows with all NaN in key fields
df = df.dropna(subset=['Authors', 'Title', 'Source title'])
print(f"[10] After key field check: {len(df)} rows")

# Remove exact duplicate rows
initial_len = len(df)
df = df.drop_duplicates()
if len(df) < initial_len:
    print(f"[11] Removed {initial_len - len(df)} duplicate rows: {len(df)} rows remain")
else:
    print(f"[11] No duplicates found: {len(df)} rows")

# ============================================================================
# FINAL ORDERING & INDEXING
# ============================================================================

# Sort by citation count (descending), then year (descending)
df = df.sort_values(['Cited by', 'Year'], ascending=[False, False])

# Add sequence number
df.insert(0, 'Sr No', range(1, len(df) + 1))

# Reset index
df = df.reset_index(drop=True)

print(f"[12] Final ordering and indexing: {len(df)} rows")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

output_file = 'Electronic-Commerce-Research_TopicModelling_Export.csv'
df.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print(f"✓ Exported to: {output_file}")
print(f"✓ Final row count: {len(df)}")
print(f"✓ Year range: {df['Year'].min()}-{df['Year'].max()}")
print(f"✓ Citation range: {df['Cited by'].min()}-{df['Cited by'].max()}")
print(f"{'='*70}\n")

# Optional: show summary stats
print("Summary Statistics:")
print(f"  Rows with abstract: {df['Abstract'].notna().sum()}")
print(f"  Rows with author keywords: {df['Author Keywords'].notna().sum()}")
print(f"  Unique sources: {df['Source title'].nunique()}")
print()

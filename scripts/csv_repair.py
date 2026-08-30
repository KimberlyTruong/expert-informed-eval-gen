"""Repair malformed model-generated CSV before batch annotation collection.

Language models sometimes emit unquoted commas in the final text column. The
repair function preserves the expected columns by merging overflow fields back
into that final column. It is used only by batch label-result collection.
"""

import csv
import io
from typing import Optional, List


def repair_csv_last_column_overflow(csv_text: str) -> Optional[str]:
    """Detect and repair CSV where last column contains unescaped commas/newlines.

    Assumes:
    - Last column is named 'example_text' or 'example text'
    - Extra fields in a row belong to the last column
    - Re-escaping/re-quoting the last column will fix the parse error

    Returns repaired CSV text, or None if repair not applicable.

    Example:
        Input (malformed, extra fields):
            column1,column2,example_text
            value1,value2,text with comma, inside

        Output (repaired):
            column1,column2,example_text
            value1,value2,"text with comma, inside"
    """

    def _strip_code_fences(text: str) -> str:
        lines_local = text.splitlines()
        if not lines_local:
            return text

        if lines_local[0].lstrip().startswith("```"):
            lines_local = lines_local[1:]

        if lines_local and lines_local[-1].strip().startswith("```"):
            lines_local = lines_local[:-1]

        return "\n".join(lines_local)

    def _unwrap_fully_quoted_lines(lines_local: List[str]) -> List[str]:
        if not lines_local:
            return lines_local

        header_candidate = lines_local[0].strip()
        if not (
            header_candidate.startswith('"') and header_candidate.endswith('"')
        ):
            return lines_local

        # Only unwrap when the header looks like a single quoted line containing commas
        # i.e., exactly two double-quote chars surrounding the content. If there are
        # multiple quoted fields (many quotes) we should not unwrap.
        if header_candidate.count('"') != 2:
            return lines_local

        if "," not in header_candidate[1:-1]:
            return lines_local

        unwrapped: List[str] = []
        for line in lines_local:
            stripped = line.strip()
            if stripped.startswith('"') and stripped.endswith('"'):
                inner = stripped[1:-1].replace('""', '"')
                unwrapped.append(inner)
            else:
                unwrapped.append(line)
        return unwrapped

    csv_text = _strip_code_fences(csv_text)
    # Split into physical lines, then join broken logical CSV rows where
    # double-quote parity is odd (a row was split by an unescaped newline
    # inside a quoted field). This helps handle stray/unbalanced quotes
    # produced by model outputs that include unescaped newlines.
    raw_lines = csv_text.splitlines()

    def _join_broken_lines(lines_local: List[str]) -> List[str]:
        joined: List[str] = []
        buf: Optional[str] = None
        for ln in lines_local:
            if buf is None:
                buf = ln
            else:
                buf = buf + "\n" + ln

            # Count double-quote characters. Escaped quotes appear as "" which
            # increments the count by 2, so parity still indicates balance.
            if buf.count('"') % 2 == 0:
                joined.append(buf)
                buf = None

        if buf is not None:
            # Leftover unbalanced buffer: append as-is so other heuristics can
            # attempt to repair it.
            joined.append(buf)

        return joined

    lines = _join_broken_lines(raw_lines)
    lines = _unwrap_fully_quoted_lines(lines)
    if not lines:
        return None

    # Parse header to find expected column count and identify last column
    header_line = lines[0]
    try:
        header_reader = csv.reader([header_line])
        headers = next(header_reader)
    except Exception:
        return None

    if not headers:
        return None

    expected_cols = len(headers)
    first_header = headers[0].strip().lower()

    repaired_lines = [header_line]

    def _has_comma_outside_parentheses(text: str) -> bool:
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                return True
        return False

    def _needs_requote_for_example_text_first(row: List[str]) -> bool:
        if first_header not in {"example text", "example_text"}:
            return False

        for value in row[1:]:
            if _has_comma_outside_parentheses(value):
                return True
        return False

    def _split_ignoring_parens(line: str, max_splits: int) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        splits = 0
        for ch in line:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1

            if ch == "," and depth == 0 and splits < max_splits:
                parts.append("".join(cur))
                cur = []
                splits += 1
            else:
                cur.append(ch)
        parts.append("".join(cur))
        return parts

    def _rsplit_ignoring_parens(line: str, max_splits: int) -> List[str]:
        parts: List[str] = []
        cur: List[str] = []
        depth = 0
        splits = 0
        for ch in reversed(line):
            if ch == ")":
                depth += 1
            elif ch == "(" and depth > 0:
                depth -= 1

            if ch == "," and depth == 0 and splits < max_splits:
                parts.append("".join(reversed(cur)))
                cur = []
                splits += 1
            else:
                cur.append(ch)

        parts.append("".join(reversed(cur)))
        return list(reversed(parts))

    for line in lines[1:]:
        if not line.strip():
            continue

        try:
            reader = csv.reader([line])
            row = next(reader)
        except Exception:
            # Try paren-aware split when csv.reader fails
            parts = _split_ignoring_parens(line, expected_cols - 1)
            if len(parts) == expected_cols:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([p.strip() for p in parts])
                repaired_lines.append(output.getvalue().rstrip("\n"))
                continue
            repaired_lines.append(line)
            continue

        if len(row) == expected_cols:
            if _needs_requote_for_example_text_first(row):
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([p.strip() for p in row])
                repaired_lines.append(output.getvalue().rstrip("\n"))
                continue
            repaired_lines.append(line)
            continue

        def _merge_into_first_column(
            parts: List[str], expected_count: int
        ) -> List[str]:
            if len(parts) <= expected_count:
                return parts

            extra = len(parts) - expected_count
            if extra <= 0:
                return parts

            first = ",".join(p.strip() for p in parts[: extra + 1])
            remainder = [p.strip() for p in parts[extra + 1 :]]
            return [first, *remainder]

        # Try paren-aware split on raw line first
        parts = _split_ignoring_parens(line, expected_cols - 1)
        if len(parts) == expected_cols:
            if first_header in {"example text", "example_text"}:
                right_split = _rsplit_ignoring_parens(line, expected_cols - 1)
                if len(right_split) == expected_cols and right_split != parts:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow([p.strip() for p in right_split])
                    repaired_lines.append(output.getvalue().rstrip("\n"))
                    continue

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([p.strip() for p in parts])
            repaired_lines.append(output.getvalue().rstrip("\n"))
            continue

        if len(parts) > expected_cols and first_header in {
            "example text",
            "example_text",
        }:
            right_split = _rsplit_ignoring_parens(line, expected_cols - 1)
            if len(right_split) == expected_cols:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow([p.strip() for p in right_split])
                repaired_lines.append(output.getvalue().rstrip("\n"))
                continue

            merged = _merge_into_first_column(parts, expected_cols)
            if len(merged) == expected_cols:
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(merged)
                repaired_lines.append(output.getvalue().rstrip("\n"))
                continue

        # If too many fields, combine based on first column name
        if len(row) > expected_cols:
            if first_header in {"example text", "example_text"}:
                right_split = _rsplit_ignoring_parens(line, expected_cols - 1)
                if len(right_split) == expected_cols:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow([p.strip() for p in right_split])
                    repaired_lines.append(output.getvalue().rstrip("\n"))
                    continue

                merged = _merge_into_first_column(row, expected_cols)
                if len(merged) == expected_cols:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(merged)
                    repaired_lines.append(output.getvalue().rstrip("\n"))
                    continue

            fixed_row = row[: expected_cols - 1]
            last_col_values = row[expected_cols - 1 :]
            last_col_text = ",".join(last_col_values)
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([*(c for c in fixed_row), last_col_text])
            repaired_lines.append(output.getvalue().rstrip("\n"))
            continue

        # If too few fields, pad
        if len(row) < expected_cols:
            padded = row + [""] * (expected_cols - len(row))
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([p.strip() for p in padded])
            repaired_lines.append(output.getvalue().rstrip("\n"))
            continue

    return "\n".join(repaired_lines)

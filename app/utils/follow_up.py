import re
from typing import List, Tuple


def extract_follow_ups(answer: str) -> Tuple[str, List[str]]:
    pattern = r"\[FOLLOW_UP\]:\s*(.+?)(?=\n|$)"
    follow_ups = re.findall(pattern, answer)
    clean_answer = re.sub(r"\n*\[FOLLOW_UP\]:\s*.+", "", answer).strip()
    return clean_answer, follow_ups[:3]

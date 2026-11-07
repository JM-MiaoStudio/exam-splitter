"""Conservative selection for optional enhanced/cloud candidates."""
import copy


def choose_assisted(original,candidates,source):
    """Preserve existing boxes; rescue abstentions only with reviewable sequences."""
    out=copy.deepcopy(original)
    if out:
        if candidates and [b['printedNumber'] for b in out]!=[b['printedNumber'] for b in candidates]:
            for b in out:
                b['unreadableParts'].append(f'{source}识别的题号与当前题框不一致，请核对是否漏题或混入作答')
        return out
    if len(candidates)<2:return []
    numbers=[b['printedNumber'] for b in candidates]
    if not all(n is None for n in numbers):
        if any(n is None or not str(n).isdigit() for n in numbers):return []
        ns=[int(n) for n in numbers]
        if ns!=list(range(ns[0],ns[0]+len(ns))):return []
    out=copy.deepcopy(candidates)
    for b in out:
        b.setdefault('regions',[copy.deepcopy(b['bbox'])])
        b['unreadableParts'].append(f'{source}补充的候选题框，请核对题目完整性与配图归属')
    return out

## LLM fine-tuning report
Processed files: 10

### before LLM
Average Line Accuracy: 81.76%  
Average Char Accuracy: 92.60%

### after LLM
#### V.0
Average Line Accuracy: 86.22%  
Average Char Accuracy: 93.73%

#### V.1
```python
# prompt_builder.py
def _lock_len_tag(conf: float | None, L: int) -> str:
    """
    基于长度与置信度给出长度锁定提示：
      - L <= 2:        HARD  （严格不改长度）
      - conf >= 0.92:  HARD
      - 0.80 <= conf < 0.92: SOFT
      - else:          NONE
    """
    if L <= 2:
        return "HARD"
    if conf is None:
        return "NONE"
    if conf >= 0.92:
        return "HARD"
    if conf >= 0.80:
        return "SOFT"
    return "NONE"
```

Average Line Accuracy: 86.43%  
Average Char Accuracy: 93.84%

#### V.2
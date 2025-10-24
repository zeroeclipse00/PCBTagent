## LLM fine-tuning report
Processed files: 10

conf_threshold: 0.95

### before LLM
Average Line Accuracy: 80.83%  
Average Char Accuracy: 92.34%

### after LLM
#### V.0
Average Line Accuracy: 85.09%  
Average Char Accuracy: 93.45%

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

Average Line Accuracy: 84.40%  
Average Char Accuracy: 92.96%  
(rollback to V0)

#### V.2

置信度调回1.1（全部纠错）

Average Line Accuracy: 83.53%  
Average Char Accuracy: 93.06%

#### V.3

更新kb

Average Line Accuracy: 84.38%  

Average Char Accuracy: 93.26%

测kb在4o和4o-mini中能读进去多少

- 二分，增加硬约束
- 结果：约240行（4o-mini）

#### V.4

- 使用4o            

Processed files: 9     
Average Line Accuracy: 87.24%  
Average Char Accuracy: 93.59%     

- 修补GPIO port的IO表达  

Average Line Accuracy: 89.96%  
Average Char Accuracy: 95.02%

- 扩充正则，允许gpio前后缀

  - 4o：

    Average Line Accuracy: 90.02%
    Average Char Accuracy: 95.05%

  - 4o-mini：

    Average Line Accuracy: 83.73%
    Average Char Accuracy: 92.76%

#### Conclusion

**4o+kb V.2**

002_01_003.txt: line_acc=97.30%, char_acc=99.60% -> 002_01_003_analysis_report_LLM.txt
014_02_001.txt: line_acc=83.21%, char_acc=94.56% -> 014_02_001_analysis_report_LLM.txt
023_01_001.txt: line_acc=84.93%, char_acc=91.13% -> 023_01_001_analysis_report_LLM.txt
023_03_001.txt: line_acc=89.01%, char_acc=91.56% -> 023_03_001_analysis_report_LLM.txt
026_01_003.txt: line_acc=86.79%, char_acc=94.25% -> 026_01_003_analysis_report_LLM.txt
026_01_004.txt: line_acc=91.94%, char_acc=96.96% -> 026_01_004_analysis_report_LLM.txt
038_01_001.txt: line_acc=86.22%, char_acc=95.34% -> 038_01_001_analysis_report_LLM.txt
045_01_001.txt: line_acc=92.58%, char_acc=97.47% -> 045_01_001_analysis_report_LLM.txt
050_01_001.txt: line_acc=90.00%, char_acc=90.91% -> 050_01_001_analysis_report_LLM.txt
070_02_001.txt: line_acc=96.84%, char_acc=99.28% -> 070_02_001_analysis_report_LLM.txt
077_01_001.txt: line_acc=65.69%, char_acc=88.66% -> 077_01_001_analysis_report_LLM.txt
094_01_001.txt: line_acc=93.38%, char_acc=98.54% -> 094_01_001_analysis_report_LLM.txt
102_01_008.txt: line_acc=90.48%, char_acc=97.00% -> 102_01_008_analysis_report_LLM.txt
153_02_001.txt: line_acc=80.99%, char_acc=93.81% -> 153_02_001_analysis_report_LLM.txt
155_02_001.txt: line_acc=83.42%, char_acc=93.65% -> 155_02_001_analysis_report_LLM.txt
158_01_006.txt: line_acc=100.00%, char_acc=100.00% -> 158_01_006_analysis_report_LLM.txt
165_01_003.txt: line_acc=81.56%, char_acc=93.54% -> 165_01_003_analysis_report_LLM.txt
177_01_001.txt: line_acc=80.00%, char_acc=95.65% -> 177_01_001_analysis_report_LLM.txt
180_01_003.txt: line_acc=76.65%, char_acc=93.55% -> 180_01_003_analysis_report_LLM.txt
197_01_002.txt: line_acc=88.82%, char_acc=97.62% -> 197_01_002_analysis_report_LLM.txt
208_01_002.txt: line_acc=86.27%, char_acc=92.37% -> 208_01_002_analysis_report_LLM.txt
213_02_001.txt: line_acc=100.00%, char_acc=100.00% -> 213_02_001_analysis_report_LLM.txt
225_01_001.txt: line_acc=87.71%, char_acc=93.83% -> 225_01_001_analysis_report_LLM.txt
227_01_001.txt: line_acc=88.24%, char_acc=90.06% -> 227_01_001_analysis_report_LLM.txt
229_01_001.txt: line_acc=81.25%, char_acc=92.23% -> 229_01_001_analysis_report_LLM.txt
237_01_003.txt: line_acc=91.95%, char_acc=96.90% -> 237_01_003_analysis_report_LLM.txt
241_03_001.txt: line_acc=65.60%, char_acc=92.47% -> 241_03_001_analysis_report_LLM.txt
269_01_001.txt: line_acc=76.97%, char_acc=87.47% -> 269_01_001_analysis_report_LLM.txt
297_01_004.txt: line_acc=78.72%, char_acc=96.74% -> 297_01_004_analysis_report_LLM.txt
297_01_005.txt: line_acc=77.59%, char_acc=97.75% -> 297_01_005_analysis_report_LLM.txt

=== Overall Summary ===
Processed files: 30
Average Line Accuracy: 85.80%
Average Char Accuracy: 94.76%

**4o+kb V.1**

002_01_003.txt: line_acc=94.59%, char_acc=98.59% -> 002_01_003_analysis_report_LLM.txt
014_02_001.txt: line_acc=84.35%, char_acc=95.19% -> 014_02_001_analysis_report_LLM.txt
023_01_001.txt: line_acc=87.67%, char_acc=92.55% -> 023_01_001_analysis_report_LLM.txt
023_03_001.txt: line_acc=85.71%, char_acc=90.62% -> 023_03_001_analysis_report_LLM.txt
026_01_003.txt: line_acc=86.79%, char_acc=94.25% -> 026_01_003_analysis_report_LLM.txt
026_01_004.txt: line_acc=91.94%, char_acc=96.96% -> 026_01_004_analysis_report_LLM.txt
038_01_001.txt: line_acc=65.31%, char_acc=84.63% -> 038_01_001_analysis_report_LLM.txt
045_01_001.txt: line_acc=91.70%, char_acc=97.36% -> 045_01_001_analysis_report_LLM.txt
050_01_001.txt: line_acc=80.00%, char_acc=87.88% -> 050_01_001_analysis_report_LLM.txt
070_02_001.txt: line_acc=94.74%, char_acc=98.80% -> 070_02_001_analysis_report_LLM.txt
077_01_001.txt: line_acc=65.69%, char_acc=88.66% -> 077_01_001_analysis_report_LLM.txt
094_01_001.txt: line_acc=94.43%, char_acc=98.76% -> 094_01_001_analysis_report_LLM.txt
102_01_008.txt: line_acc=90.48%, char_acc=97.00% -> 102_01_008_analysis_report_LLM.txt
153_02_001.txt: line_acc=85.21%, char_acc=94.76% -> 153_02_001_analysis_report_LLM.txt
155_02_001.txt: line_acc=82.90%, char_acc=93.07% -> 155_02_001_analysis_report_LLM.txt
158_01_006.txt: line_acc=100.00%, char_acc=100.00% -> 158_01_006_analysis_report_LLM.txt
165_01_003.txt: line_acc=81.25%, char_acc=93.28% -> 165_01_003_analysis_report_LLM.txt
177_01_001.txt: line_acc=80.00%, char_acc=95.65% -> 177_01_001_analysis_report_LLM.txt
180_01_003.txt: line_acc=74.45%, char_acc=92.73% -> 180_01_003_analysis_report_LLM.txt
197_01_002.txt: line_acc=90.13%, char_acc=97.86% -> 197_01_002_analysis_report_LLM.txt
208_01_002.txt: line_acc=85.62%, char_acc=92.17% -> 208_01_002_analysis_report_LLM.txt
213_02_001.txt: line_acc=100.00%, char_acc=100.00% -> 213_02_001_analysis_report_LLM.txt
225_01_001.txt: line_acc=87.71%, char_acc=93.83% -> 225_01_001_analysis_report_LLM.txt
227_01_001.txt: line_acc=86.93%, char_acc=89.66% -> 227_01_001_analysis_report_LLM.txt
229_01_001.txt: line_acc=82.95%, char_acc=92.64% -> 229_01_001_analysis_report_LLM.txt
237_01_003.txt: line_acc=90.23%, char_acc=96.51% -> 237_01_003_analysis_report_LLM.txt
241_03_001.txt: line_acc=84.00%, char_acc=92.47% -> 241_03_001_analysis_report_LLM.txt
269_01_001.txt: line_acc=80.00%, char_acc=88.12% -> 269_01_001_analysis_report_LLM.txt
297_01_004.txt: line_acc=74.47%, char_acc=96.27% -> 297_01_004_analysis_report_LLM.txt
297_01_005.txt: line_acc=70.69%, char_acc=95.91% -> 297_01_005_analysis_report_LLM.txt

=== Overall Summary ===
Processed files: 30
Average Line Accuracy: 85.00%
Average Char Accuracy: 94.21%

**4o-mini+kb V.1**

002_01_003.txt: line_acc=86.49%, char_acc=97.38% -> 002_01_003_analysis_report_LLM.txt
014_02_001.txt: line_acc=76.34%, char_acc=93.38% -> 014_02_001_analysis_report_LLM.txt
023_01_001.txt: line_acc=84.93%, char_acc=91.13% -> 023_01_001_analysis_report_LLM.txt
023_03_001.txt: line_acc=85.71%, char_acc=90.62% -> 023_03_001_analysis_report_LLM.txt
026_01_003.txt: line_acc=86.79%, char_acc=94.25% -> 026_01_003_analysis_report_LLM.txt
026_01_004.txt: line_acc=91.94%, char_acc=96.96% -> 026_01_004_analysis_report_LLM.txt
038_01_001.txt: line_acc=62.76%, char_acc=84.26% -> 038_01_001_analysis_report_LLM.txt
045_01_001.txt: line_acc=64.63%, char_acc=94.93% -> 045_01_001_analysis_report_LLM.txt
050_01_001.txt: line_acc=90.00%, char_acc=90.91% -> 050_01_001_analysis_report_LLM.txt
070_02_001.txt: line_acc=85.26%, char_acc=96.65% -> 070_02_001_analysis_report_LLM.txt
077_01_001.txt: line_acc=64.96%, char_acc=88.52% -> 077_01_001_analysis_report_LLM.txt
094_01_001.txt: line_acc=89.55%, char_acc=97.74% -> 094_01_001_analysis_report_LLM.txt
102_01_008.txt: line_acc=90.48%, char_acc=97.00% -> 102_01_008_analysis_report_LLM.txt
153_02_001.txt: line_acc=75.35%, char_acc=92.22% -> 153_02_001_analysis_report_LLM.txt
155_02_001.txt: line_acc=82.38%, char_acc=93.30% -> 155_02_001_analysis_report_LLM.txt
158_01_006.txt: line_acc=77.78%, char_acc=95.65% -> 158_01_006_analysis_report_LLM.txt
165_01_003.txt: line_acc=65.00%, char_acc=91.34% -> 165_01_003_analysis_report_LLM.txt
177_01_001.txt: line_acc=80.00%, char_acc=95.65% -> 177_01_001_analysis_report_LLM.txt
180_01_003.txt: line_acc=69.60%, char_acc=91.83% -> 180_01_003_analysis_report_LLM.txt
197_01_002.txt: line_acc=89.47%, char_acc=97.62% -> 197_01_002_analysis_report_LLM.txt
208_01_002.txt: line_acc=83.66%, char_acc=91.39% -> 208_01_002_analysis_report_LLM.txt
213_02_001.txt: line_acc=100.00%, char_acc=100.00% -> 213_02_001_analysis_report_LLM.txt
225_01_001.txt: line_acc=86.35%, char_acc=93.33% -> 225_01_001_analysis_report_LLM.txt
227_01_001.txt: line_acc=84.31%, char_acc=88.84% -> 227_01_001_analysis_report_LLM.txt
229_01_001.txt: line_acc=82.95%, char_acc=92.64% -> 229_01_001_analysis_report_LLM.txt
237_01_003.txt: line_acc=90.23%, char_acc=96.51% -> 237_01_003_analysis_report_LLM.txt
241_03_001.txt: line_acc=84.80%, char_acc=92.68% -> 241_03_001_analysis_report_LLM.txt
269_01_001.txt: line_acc=78.18%, char_acc=87.73% -> 269_01_001_analysis_report_LLM.txt
297_01_004.txt: line_acc=78.72%, char_acc=97.20% -> 297_01_004_analysis_report_LLM.txt
297_01_005.txt: line_acc=67.24%, char_acc=95.50% -> 297_01_005_analysis_report_LLM.txt

=== Overall Summary ===
Processed files: 30
Average Line Accuracy: 81.20%
Average Char Accuracy: 93.57%

**4o-mini+kb V.2**

002_01_003.txt: line_acc=93.24%, char_acc=98.39% -> 002_01_003_analysis_report_LLM.txt
014_02_001.txt: line_acc=81.68%, char_acc=93.93% -> 014_02_001_analysis_report_LLM.txt
023_01_001.txt: line_acc=83.56%, char_acc=90.43% -> 023_01_001_analysis_report_LLM.txt
023_03_001.txt: line_acc=84.62%, char_acc=90.31% -> 023_03_001_analysis_report_LLM.txt
026_01_003.txt: line_acc=86.79%, char_acc=94.25% -> 026_01_003_analysis_report_LLM.txt
026_01_004.txt: line_acc=91.94%, char_acc=96.96% -> 026_01_004_analysis_report_LLM.txt
038_01_001.txt: line_acc=70.92%, char_acc=89.92% -> 038_01_001_analysis_report_LLM.txt
045_01_001.txt: line_acc=90.83%, char_acc=96.59% -> 045_01_001_analysis_report_LLM.txt
050_01_001.txt: line_acc=90.00%, char_acc=90.91% -> 050_01_001_analysis_report_LLM.txt
070_02_001.txt: line_acc=85.26%, char_acc=96.65% -> 070_02_001_analysis_report_LLM.txt
077_01_001.txt: line_acc=57.66%, char_acc=86.17% -> 077_01_001_analysis_report_LLM.txt
094_01_001.txt: line_acc=90.59%, char_acc=97.81% -> 094_01_001_analysis_report_LLM.txt
102_01_008.txt: line_acc=85.71%, char_acc=95.51% -> 102_01_008_analysis_report_LLM.txt
153_02_001.txt: line_acc=77.46%, char_acc=92.78% -> 153_02_001_analysis_report_LLM.txt
155_02_001.txt: line_acc=82.38%, char_acc=93.30% -> 155_02_001_analysis_report_LLM.txt
158_01_006.txt: line_acc=100.00%, char_acc=100.00% -> 158_01_006_analysis_report_LLM.txt
165_01_003.txt: line_acc=71.88%, char_acc=93.02% -> 165_01_003_analysis_report_LLM.txt
177_01_001.txt: line_acc=71.43%, char_acc=92.66% -> 177_01_001_analysis_report_LLM.txt
180_01_003.txt: line_acc=73.57%, char_acc=92.65% -> 180_01_003_analysis_report_LLM.txt
197_01_002.txt: line_acc=90.13%, char_acc=97.86% -> 197_01_002_analysis_report_LLM.txt
208_01_002.txt: line_acc=85.62%, char_acc=92.17% -> 208_01_002_analysis_report_LLM.txt
213_02_001.txt: line_acc=100.00%, char_acc=100.00% -> 213_02_001_analysis_report_LLM.txt
225_01_001.txt: line_acc=86.69%, char_acc=93.46% -> 225_01_001_analysis_report_LLM.txt
227_01_001.txt: line_acc=86.93%, char_acc=89.66% -> 227_01_001_analysis_report_LLM.txt
229_01_001.txt: line_acc=80.68%, char_acc=91.69% -> 229_01_001_analysis_report_LLM.txt
237_01_003.txt: line_acc=83.91%, char_acc=94.18% -> 237_01_003_analysis_report_LLM.txt
241_03_001.txt: line_acc=83.20%, char_acc=92.05% -> 241_03_001_analysis_report_LLM.txt
269_01_001.txt: line_acc=70.91%, char_acc=84.73% -> 269_01_001_analysis_report_LLM.txt
297_01_004.txt: line_acc=74.47%, char_acc=96.74% -> 297_01_004_analysis_report_LLM.txt
297_01_005.txt: line_acc=65.52%, char_acc=95.30% -> 297_01_005_analysis_report_LLM.txt

=== Overall Summary ===
Processed files: 30
Average Line Accuracy: 82.59%
Average Char Accuracy: 93.67%

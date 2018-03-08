import enchant
import pandas as pd
from enchant.checker import SpellChecker
from enchant.checker.CmdLineChecker import CmdLineChecker


df = pd.read_csv("../train_translation.csv")
d = enchant.Dict("en_US")

d.check(df['text'].iloc[0:10000, :])
d.check("love")
d.suggest("I love you")
test = df['text'][1]
d.check(test)

chen = enchant.checker.SpellChecker("en_US")
chen.set_text(test)
for err in chen:
    print(err.word)
    sug = err.suggest()[0]
    err.replace(sug)

c = chen.get_text()#returns corrected text
print(c)
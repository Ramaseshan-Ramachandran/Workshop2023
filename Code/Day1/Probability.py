from collections import Counter

sentence = "Language models have found extensive applications across diverse industries. Social media platforms utilize them to amplify user voices and improve engagement. Customer support systems leverage language models to provide efficient and accurate assistance, ensuring prompt query resolution."
words = sentence.split()
word_counts = Counter(words)

table = "\\begin{tabular}{|c|c|}\n\\hline\nWord & Count \\\\\n\\hline\n"
for word, count in word_counts.items():
    table += f"{word} & {count}\\\\\n\\hline\n"
table += "\\end{tabular}"
print(table)

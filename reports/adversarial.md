# Adversarial Detection Rate by Technique

_Fraction of obfuscated payloads per bypass technique scored malicious (threshold 0.5), from `datasets/adversarial_testset.csv`._

| technique | baseline | cnn | distilbert |
|---|---|---|---|
| ascii-function | 1.000 | 1.000 | 1.000 |
| between-operator | 1.000 | 1.000 | 1.000 |
| case-mix-and-comment | 1.000 | 1.000 | 1.000 |
| case-mixing | 1.000 | 1.000 | 1.000 |
| case-mixing-stacked | 1.000 | 1.000 | 1.000 |
| char-function | 1.000 | 1.000 | 1.000 |
| dash-dash-dash-comment | 1.000 | 1.000 | 1.000 |
| double-url-encode | 0.000 | 0.333 | 0.000 |
| float-literal | 1.000 | 1.000 | 1.000 |
| gbk-multibyte-quote | 1.000 | 1.000 | 1.000 |
| hash-comment | 1.000 | 1.000 | 1.000 |
| hex-cast | 1.000 | 1.000 | 1.000 |
| hex-literal | 1.000 | 1.000 | 1.000 |
| in-operator | 1.000 | 1.000 | 1.000 |
| inline-comment | 1.000 | 1.000 | 1.000 |
| inline-comment-lower | 1.000 | 1.000 | 1.000 |
| like-operator | 1.000 | 1.000 | 1.000 |
| mysql-versioned-comment | 1.000 | 1.000 | 1.000 |
| no-space-parens | 1.000 | 1.000 | 1.000 |
| not-operator | 1.000 | 1.000 | 1.000 |
| null-byte | 0.500 | 0.500 | 1.000 |
| overlong-utf8-quote | 1.000 | 1.000 | 1.000 |
| regexp-operator | 1.000 | 1.000 | 1.000 |
| rlike-operator | 1.000 | 1.000 | 1.000 |
| scientific-notation | 1.000 | 1.000 | 1.000 |
| split-keyword | 1.000 | 1.000 | 1.000 |
| string-concat | 1.000 | 1.000 | 1.000 |
| string-concat-mssql | 1.000 | 1.000 | 1.000 |
| string-concat-oracle | 1.000 | 1.000 | 1.000 |
| unhex-function | 1.000 | 1.000 | 1.000 |
| unicode-string-prefix | 1.000 | 1.000 | 1.000 |
| url-encode-basic | 0.000 | 0.000 | 0.000 |
| url-encode-comment | 1.000 | 1.000 | 1.000 |
| url-encode-paren | 0.000 | 0.000 | 0.000 |
| url-encode-space | 0.000 | 0.000 | 1.000 |
| whitespace-as-newline | 0.000 | 1.000 | 1.000 |
| whitespace-as-plus | 0.000 | 1.000 | 1.000 |
| whitespace-as-tab | 0.000 | 1.000 | 1.000 |
| whitespace-comment | 1.000 | 1.000 | 1.000 |

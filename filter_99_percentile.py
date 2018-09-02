import pandas
import sys

from util.numbers import normalize_numbers


df = pandas.read_csv(sys.argv[1], encoding='utf-8')

# exclude super long outliers
_99 = df['wav_filesize'].quantile(.99)

# filter out invalid transcripts with HTML data or dates, and any null transcripts
df = df[~(df['transcript'].str.contains('<') | df['transcript'].str.contains('/') | df['transcript'].isnull())]

# filter shorter than 0.5 seconds
df = df[~(df['wav_filesize'] < 16044)]

# normalize numbers
df['transcript'] = df['transcript'].apply(normalize_numbers)

# remove punctuation
df['transcript'] = df['transcript'].str.replace(r'[_.,?!`:;\[\]+\(\)\'"]', '')

df['transcript'] = df['transcript'].str.replace('-', ' ')

# expand ampersand
df['transcript'] = df['transcript'].str.replace('&', 'and')

df[df['wav_filesize'] <= _99].to_csv(sys.argv[2], index=False)

print('{} -> {}'.format(sys.argv[1], sys.argv[2]))

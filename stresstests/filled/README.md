# Filled elicitation sheets

One file per speaker, named `<speaker-id>.tsv`, copied from
`../elicitation_sheet.tsv` with the slot columns filled in.

Y = a vowel is pronounced at that numbered blank
N = no vowel there
V = genuinely variable for this speaker

Speakers must not see each other's sheets, and must not see any predicted
pronunciation. Two speakers is the minimum; three lets a majority break ties.

Merge with:

    python -m src.g2p.elicit merge stresstests/hindi_schwa_set.tsv \
        "stresstests/filled/*.tsv"

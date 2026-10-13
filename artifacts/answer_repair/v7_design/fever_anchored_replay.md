# FEVER v7 anchored decision: offline replay of v6r1 traces

Same MNLI windows and probabilities as v6r1; only the decision rule changes. Development evidence on the cohorts the rule was designed against, not held-out.

| cohort | n | initial correct | v6 final | v7 final | v6 fixed/regressed | v7 fixed/regressed | finals changed v6→v7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reviewed | 14 | 4 | 7 | 10 | 3/0 | 6/0 | 3 |
| dev50 | 50 | 45 | 44 | 46 | 1/2 | 1/0 | 2 |

## Samples whose final answer changed between v6 and v7

- reviewed `146639` In 2003, Eva Green debuted in film. | initial REFUTES | v6 REFUTES (nli_conflict, still_wrong) → v7 SUPPORTS (repaired, fixed) | reference SUPPORTS
  - excluded R4 `The_Big_Green` E=0.0003 C=0.9972 missing anchors ['2003', 'eva']
  - excluded R5 `Adam_Green's_Aladdin` E=0.0046 C=0.9449 missing anchors ['2003', 'eva']
- reviewed `161905` In 1974 Alan Shepard retired from the US Navy. | initial REFUTES | v6 REFUTES (nli_conflict, still_wrong) → v7 SUPPORTS (repaired, fixed) | reference SUPPORTS
  - excluded R4 `John_P._Weinel` E=0.0051 C=0.9828 missing anchors ['alan', 'shepard']
  - excluded R5 `Timothy_A._Kinnan` E=0.0011 C=0.9952 missing anchors ['1974', 'shepard', 'navy']
- reviewed `71180` Babur was a Mughal emperor. | initial REFUTES | v6 REFUTES (nli_conflict, still_wrong) → v7 SUPPORTS (repaired, fixed) | reference SUPPORTS
  - excluded R4 `Bidar_Bakht` E=0.0082 C=0.9822 missing anchors ['babur']
- dev50 `140820` Men in Black II is a film from America. | initial SUPPORTS | v6 REFUTES (repaired, regressed) → v7 SUPPORTS (nli_uncertain, correct_preserved) | reference SUPPORTS
  - excluded R2 `Black_-LRB-2015_Belgian_film-RRB-` E=0.0005 C=0.9792 missing anchors ['men', 'ii', 'america']
  - excluded R4 `In_America_-LRB-film-RRB-` E=0.0057 C=0.9592 missing anchors ['men', 'black', 'ii']
- dev50 `33982` Henry Cavill starred in two Superman films. | initial SUPPORTS | v6 REFUTES (repaired, regressed) → v7 SUPPORTS (nli_uncertain, correct_preserved) | reference SUPPORTS
  - excluded R5 `Henry's_Crime` E=0.0071 C=0.9253 missing anchors ['cavill', 'superman']

## Samples where v7 excluded a decisive window without changing the final answer

- reviewed `156493` Daggering is not a type of dance. | repaired → repaired | excluded R4 Ceremonial_dance
- reviewed `193910` Bea Arthur was born under a different name. | kept → nli_uncertain | excluded R4 Bea_Wain
- dev50 `40755` Touchscreens are only used in computers. | kept → kept | excluded R5 Handheld_PC
- dev50 `137498` Danger UXB is from 1980. | kept → nli_uncertain | excluded R2 UN_-LRB-band-RRB-, R3 Blast_-LRB-American_band-RRB-, R4 Con_Brio,_Inc., R5 UXUS
- dev50 `159932` Christa McAuliffe taught social studies. | kept → nli_uncertain | excluded R4 Mary_Mackenzie_-LRB-mezzo-soprano-RRB-
- dev50 `59067` Tatum O'Neal spent her whole life single. | kept → nli_uncertain | excluded R1 Singles_-LRB-1992_film-RRB-, R2 Shaunie_O'Neal, R5 Beyoncé_discography, R5 Beyoncé_discography
- dev50 `226143` Richard Dawkins has yet to appear on the radio. | kept → kept | excluded R3 Peter_Dawkins_-LRB-writer-RRB-
- dev50 `28381` Justine Bateman is a poet. | kept → nli_uncertain | excluded R1 Sarah_Holland-Batt, R3 Citizen-COLON-_An_American_Lyric, R5 Beth_Bachmann
- dev50 `174027` The Endless River is Pink Floyd's reunion tour. | kept → kept | excluded R4 The_Wall_Tour_-LRB-1980–81-RRB-
- dev50 `42213` John Deighton had no choice but to pursue other lines of work. | kept → nli_uncertain | excluded R2 John_Bowdler
- dev50 `42998` Richard Fortus is a Guns N' Roses member. | nli_conflict → kept | excluded R5 List_of_Guns_N'_Roses_members
- dev50 `137715` Winter's Tale was published in 1970. | kept → kept | excluded R4 The_Winter_King_-LRB-novel-RRB-, R5 The_Snowman
- dev50 `171625` Dave Gibbons was born at a Planet Fitness. | kept → nli_uncertain | excluded R1 World_Gym, R2 Dave_Palumbo, R3 Mark_Holyoake, R4 Marcus_Antebi, R5 Dave_Schultz_-LRB-wrestling-RRB-
- dev50 `98655` Faye Resnick is a person. | nli_conflict → kept | excluded R2 Faye_Grant
- dev50 `222031` Stuart Rosenberg refused to direct Brubaker. | kept → kept | excluded R3 Betrayed_-LRB-1988_film-RRB-, R4 The_Black_Gestapo, R5 Scott_Rosenberg
- dev50 `122252` The Mirny (sloop-of-war) was in expedition. | nli_conflict → kept | excluded R2 Conflict-class_sloop, R3 HMS_Royal_George_-LRB-1809-RRB-
- dev50 `108201` Kate Walsh was on a tour of duty in 1947. | kept → nli_uncertain | excluded R2 Judy_Walsh, R4 Monica_Milne
- dev50 `222650` A United Kingdom was directed by Amma Asante. | nli_conflict → nli_conflict | excluded R3 This_Is_England
- dev50 `35822` The Icelandic Coast Guard is responsible for Iceland's maritime search and destroy. | kept → nli_uncertain | excluded R4 Greenland_Patrol
- dev50 `222662` A United Kingdom is a biographical romantic drama play. | kept → nli_uncertain | excluded R2 Great_Britain_-LRB-play-RRB-
- dev50 `211290` The Closer was a cable comedy. | kept → kept | excluded R4 The_Cable_Guy
- dev50 `227361` Giada at Home aired on the Food Network beginning in 2008. | nli_conflict → kept | excluded R5 Guerrilla_Gourmet
- dev50 `101839` The Mighty Ducks was only produced by a Canadian film producer. | kept → kept | excluded R5 Duck_Studios
- dev50 `70359` Guns N' Roses started in 1995. | kept → kept | excluded R5 List_of_Guns_N'_Roses_members
- dev50 `222026` Brubaker is a solely Australian film. | kept → kept | excluded R1 Australia_-LRB-2008_film-RRB-, R4 Mortdecai_-LRB-film-RRB-, R5 Ned_Kelly_-LRB-2003_film-RRB-

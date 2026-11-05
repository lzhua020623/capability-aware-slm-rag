# E new-output human review order

Ordering is based only on frozen metrics and whether the answer changed, and does not represent a human conclusion. 541 results, 536 unique dataset/sample_id; each batch is kept separately.

Review the first 43 changes first (18 metric regressions, 13 metric improvements, 12 other changes); the remaining 498 results that kept the answer still await full evidence-support review.

For the same item, the human reviewer must judge the initial answer and the final answer separately on correctness and evidence support, four items in total; record the reviewer and the rationale.
Write confirmations into pair_review in review_priority.jsonl, and set status=completed when done; the four boolean fields must not be empty.
The original pending_semantic_pairs.jsonl is kept as an immutable audit base record. Do not use the frozen references to auto-fill human fields.

Existing correctness judgments must also be rechecked under the same criteria. Citations taken verbatim from the source text only prove the source is legitimate; returning the original answer does not mean the answer has been verified.

## 1. FEVER / heldout / 126292

Priority reason: frozen_metric_regression; result fingerprint: `a5d2381304c29ae136088cd7400d25ec931c3bb0ffdc00db9af35714bed8ea38`.

Question: Camden, New Jersey is a county.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 2. FEVER / heldout / 14448

Priority reason: frozen_metric_regression; result fingerprint: `7117a5f59ba0b48be66943865b19b186519ed5b4738db10f26e310b526405308`.

Question: Sexual slavery is a reason for human trafficking.

Initial: SUPPORTS

Final: REFUTES

Human status: pending; neither correctness nor support has been judged.

## 3. FEVER / heldout / 164986

Priority reason: frozen_metric_regression; result fingerprint: `371a0da97836b0376460ab136efcaa22e9b53f383773f2810d177d410782c429`.

Question: Polar bears are classified as endangered.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 4. FEVER / heldout / 49192

Priority reason: frozen_metric_regression; result fingerprint: `dc2111f0dbe8384b2021beac7390534955d36ad256c95d7412e83ce4ab9b8738`.

Question: The Armenian Genocide rarely was the extermination of Armenians who were mostly Ottoman citizens.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 5. NQ / heldout / -1145193137828819774

Priority reason: frozen_metric_regression; result fingerprint: `1554248420ddb3cf979d9d5e4942d6bac46e92553dee20d6e6bb01da05c27082`.

Question: who plays the grandmother in game of thrones

Initial: Rigg

Final: Lady Olenna Tyrell

Human status: pending; neither correctness nor support has been judged.

## 6. NQ / dev50 / -1450346568093066821

Priority reason: frozen_metric_regression; result fingerprint: `52fe64f765b5088f9c2bfb01faf35c8422c6fda77633ca5043b15e61d16df802`.

Question: where was because i could not stop for death published

Initial: Poems: Series 1

Final: in Poems : Series 1 in 1890

Human status: pending; neither correctness nor support has been judged.

## 7. NQ / heldout / -2324904500814145665

Priority reason: frozen_metric_regression; result fingerprint: `3e78faa1cb9c4bab0137eabfe517d9d31769028a9d9b4fc4dc2a2ea029b306da`.

Question: when was south african youth day first celebrated

Initial: 1976

Final: 16 June

Human status: pending; neither correctness nor support has been judged.

## 8. NQ / heldout / -3719623571078658627

Priority reason: frozen_metric_regression; result fingerprint: `38d03d14f364a87aa9e3edb8e4dc69242555082eaaa90f977fa44a37df6062b4`.

Question: what does rt mean on a dodge car

Initial: Road/Track

Final: Road / Track

Human status: pending; neither correctness nor support has been judged.

## 9. NQ / heldout / -5788046732019278353

Priority reason: frozen_metric_regression; result fingerprint: `02f3a4dcafe2bb11661aa7a01f0ec519d8d703034b8623fa648ef4e7b1d0a0b9`.

Question: who talks for belle in beauty and the beast

Initial: Paige O'Hara

Final: American actress and singer Paige O'Hara

Human status: pending; neither correctness nor support has been judged.

## 10. NQ / heldout / -7524627393810362735

Priority reason: frozen_metric_regression; result fingerprint: `5e01ce4a43e8089732fd49a8a72a8afe673cca1ee36b4af6c17f1f5b80fcf178`.

Question: when did men's curling start in the olympics

Initial: 1998

Final: 1924

Human status: pending; neither correctness nor support has been judged.

## 11. NQ / heldout / -7583855073136179893

Priority reason: frozen_metric_regression; result fingerprint: `b34a797bad38c387c8147db90aa5036aeb8a4cb22633a299687fd46c11103775`.

Question: when did david akers kick the 63 yard field goal

Initial: September 9, 2012

Final: On September 9 , 2012

Human status: pending; neither correctness nor support has been judged.

## 12. NQ / heldout / -7778233237890403173

Priority reason: frozen_metric_regression; result fingerprint: `085c635910988e07485dded152641c4180e5a9235edf5cd191bc37ad2d2acb8d`.

Question: where did they film the show the crossing

Initial: British Columbia, Canada

Final: coastal areas of British Columbia and in the city of Vancouver , British Columbia

Human status: pending; neither correctness nor support has been judged.

## 13. NQ / heldout / -8421690753817748624

Priority reason: frozen_metric_regression; result fingerprint: `ce9dd8cc3092df8fef6636d7e156f548809cd4d5704dbde53bf834b291eb7e02`.

Question: where is the heart in the human body

Initial: between the lungs, in the middle compartment of the chest

Final: in the middle mediastinum , at the level of thoracic vertebrae T5 - T8

Human status: pending; neither correctness nor support has been judged.

## 14. NQ / dev50 / -8825841420882619284

Priority reason: frozen_metric_regression; result fingerprint: `84bc326d9d47d374d9e90480f3ce3c5b17ddfe6eb6df427d75286ac278e43d06`.

Question: where's the tv show the crossing filmed

Initial: British Columbia, Canada

Final: coastal areas of British Columbia and in the city of Vancouver , British Columbia

Human status: pending; neither correctness nor support has been judged.

## 15. NQ / heldout / 2776741854318333179

Priority reason: frozen_metric_regression; result fingerprint: `a71af8b2cbbcf093e7b1cd26f34b1b5efcb52435edb93bac9f9a61e0ee131652`.

Question: who sings only love can break your heart

Initial: Neil Young

Final: Saint Etienne

Human status: pending; neither correctness nor support has been judged.

## 16. NQ / heldout / 3181001754120908131

Priority reason: frozen_metric_regression; result fingerprint: `547c45c889ed6ae1a9230a954d4dad557eb58ece38fb1e2d40e952098aed3360`.

Question: india's first arctic research station set up in 2008 what is the name

Initial: Himadri Station

Final: Himadri

Human status: pending; neither correctness nor support has been judged.

## 17. NQ / heldout / 3630714342371225513

Priority reason: frozen_metric_regression; result fingerprint: `b0d5229adb1c643997c79d6d7d78c87f8b6f6859b07278f3761185b218cfd746`.

Question: where did the term spirits for alcohol come from

Initial: Middle Eastern alchemy

Final: spirit of the original material

Human status: pending; neither correctness nor support has been judged.

## 18. NQ / dev50 / 5898512601992001509

Priority reason: frozen_metric_regression; result fingerprint: `f5789ea704a394db4a13da24cec202eba72852209e725d06b5b7f1c04185e732`.

Question: who did the united states fight in the war of 1812

Initial: the United Kingdom

Final: the United States , the United Kingdom , and their respective allies

Human status: pending; neither correctness nor support has been judged.

## 19. FEVER / reviewed / 146639

Priority reason: frozen_metric_gain; result fingerprint: `0dfc1acb9cd422a01e11b8143d8d2153e7da05916be079316ce44d44008c7aa6`.

Question: In 2003, Eva Green debuted in film.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 20. FEVER / reviewed / 156493

Priority reason: frozen_metric_gain; result fingerprint: `df0ab6e1c950ab263dae533ef28aa0e09007e81ee7be6a2b9d86a3e770e060b5`.

Question: Daggering is not a type of dance.

Initial: SUPPORTS

Final: REFUTES

Human status: pending; neither correctness nor support has been judged.

## 21. FEVER / reviewed / 161905

Priority reason: frozen_metric_gain; result fingerprint: `409647d9e8a19d711888554f4bd2fe66d8ae1b7505df8bd7ca461b5343b74067`.

Question: In 1974 Alan Shepard retired from the US Navy.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 22. FEVER / reviewed / 203615

Priority reason: frozen_metric_gain; result fingerprint: `0893b1cc3d8e8c9fa06f9190c30d112d51876a161ca5ce399429c0d063d1315b`.

Question: The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 23. FEVER / reviewed / 71180

Priority reason: frozen_metric_gain; result fingerprint: `0edf020f516ecea35bff6195aca9f27577f77a6278585434c71440f64502193f`.

Question: Babur was a Mughal emperor.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 24. FEVER / reviewed / 72578

Priority reason: frozen_metric_gain; result fingerprint: `9c3d56dadec77b7b00683e29b78fb737f681a64d9a2c1e3ddd669a9e9a8179c2`.

Question: The Greek language has speakers in Albania.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 25. FEVER / heldout / 104922

Priority reason: frozen_metric_gain; result fingerprint: `539fda5c8077c7ea14d85a03ef8d8819cea48442f9bc3a881f95aa4fc49a0939`.

Question: Oscar Isaac did not act as the supervillain Apocalypse.

Initial: SUPPORTS

Final: REFUTES

Human status: pending; neither correctness nor support has been judged.

## 26. FEVER / dev50 / 203615

Priority reason: frozen_metric_gain; result fingerprint: `d013f2d70808e98175a574ddceb98748dedb7b83fb2166e4819fb1db877efed2`.

Question: The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

Initial: REFUTES

Final: SUPPORTS

Human status: pending; neither correctness nor support has been judged.

## 27. NQ / reviewed / -2905696611778901658

Priority reason: frozen_metric_gain; result fingerprint: `aee1d7bb6dadd74e42d35fe5ff862a70c459bf40f23813218c09e2c25f4c5b00`.

Question: who sang oh annie i not your daddy

Initial: Annie, I'm Not Your Daddy

Final: Kid Creole and the Coconuts

Human status: pending; neither correctness nor support has been judged.

## 28. NQ / dev50 / -2905696611778901658

Priority reason: frozen_metric_gain; result fingerprint: `897598341ed2a0b95eb38256c00c423d0eeea337bb2a4430f27edb003b7da827`.

Question: who sang oh annie i not your daddy

Initial: Annie, I'm Not Your Daddy

Final: Kid Creole and the Coconuts

Human status: pending; neither correctness nor support has been judged.

## 29. NQ / dev50 / 448742752032961281

Priority reason: frozen_metric_gain; result fingerprint: `a89341acc556f916187219bb254502dcbe4820670bd84f0a9529f0d63eda24f3`.

Question: why god save the queen and not king

Initial: gender of the reigning monarch

Final: depending on the gender of the reigning monarch

Human status: pending; neither correctness nor support has been judged.

## 30. NQ / heldout / 5320268832129970522

Priority reason: frozen_metric_gain; result fingerprint: `08a28b85e9c856c5bf3bdf9b6f5568f4749b8b7e6f23084a1a6d22c9e9d057ae`.

Question: what is the number of basic units in the international system of units

Initial: 7

Final: seven

Human status: pending; neither correctness nor support has been judged.

## 31. NQ / heldout / 6453011233363353306

Priority reason: frozen_metric_gain; result fingerprint: `8cca4def02b98ef625208511e3d4f6654fe0392d32653cf046e6b4e669b326c0`.

Question: the heart muscle is stimulated to contract by electrical impulses which are generated where

Initial: SA node

Final: the sinoatrial node

Human status: pending; neither correctness nor support has been judged.

## 32. NQ / reviewed / 2326758035558088175

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `999e11bf23b2e68c5be1d74dcb752981ffa79c9e47a534a41bd38716b715c538`.

Question: what is the meaning of to have a bias

Initial: prejudice in favor of or against one thing, person, or group compared with another

Final: prejudice in favour of or against one thing , person , or group compared with another

Human status: pending; neither correctness nor support has been judged.

## 33. NQ / heldout / -19630582826733372

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `813ad447fb515d44a9dadb1b78aae0094a342d7fdcc307e651af43843fb2f732`.

Question: where is the taurus mountains located on a map

Initial: southern Turkey

Final: in southern Turkey

Human status: pending; neither correctness nor support has been judged.

## 34. NQ / dev50 / -2743701347497238377

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `4e4e82e4194ed8da2c338dcd07eab634840ba50dbcea983182cf8b0e431237d2`.

Question: how oxygenated blood returns to the heart from the lungs

Initial: pulmonary veins

Final: through pulmonary veins

Human status: pending; neither correctness nor support has been judged.

## 35. NQ / heldout / -4170119894384300076

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `e898ddffd16d456bab3b8dfe5a77dc4d97d329de4e0db2330bc90b8dfe4269df`.

Question: who do you play as in dragon age origins

Initial: the player-controlled character

Final: the player - controlled character

Human status: pending; neither correctness nor support has been judged.

## 36. NQ / heldout / -5248648953323441399

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `92f9364893d6588bebf80e158cfd71ed3f0f2332e3810e6bdfb9d6db2aa511f4`.

Question: when did god save the queen became the national anthem

Initial: 1974

Final: 19 April 1984

Human status: pending; neither correctness nor support has been judged.

## 37. NQ / dev50 / -544805225908484128

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `62ab2e4671bdb9645f78d17bbc74fc7eabaa7f333dec9790c23145ef5b404eee`.

Question: is greenland part of europe or north america

Initial: North America

Final: an autonomous constituent country within the Kingdom of Denmark

Human status: pending; neither correctness nor support has been judged.

## 38. NQ / heldout / -6081006991734955423

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `d939893a3041dec5a75d8d323ff9d398b9bfb2b1ce1643415b0bc0b3ae002e93`.

Question: where did the battle of bonhomme richard take place

Initial: Flamborough Head, Yorkshire

Final: off Flamborough Head , Yorkshire

Human status: pending; neither correctness nor support has been judged.

## 39. NQ / heldout / -8299469566867123031

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `32e760d2b0cc8ea222f5c31dc9d9cd6b1ffb63257bf39e52c7a6720bc7d4d0fb`.

Question: what is the name of india pakistan border

Initial: International Border (IB)

Final: India and Pakistan Border , known locally as the International Border ( IB )

Human status: pending; neither correctness nor support has been judged.

## 40. NQ / dev50 / 1012541351190124000

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `94fa5c2376fcc1859e84547440ee04082c82725131a2cbefe5e8ddec08bf23f4`.

Question: what is the new touchback rule in the nfl

Initial: touchback on the 25-yard line

Final: the 25 - yard line

Human status: pending; neither correctness nor support has been judged.

## 41. NQ / dev50 / 2178109984531253647

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `a25beffc9bc37c6e5ebaaabb5525bb127de24888556ac695a99cc0e6c787712f`.

Question: if i die before i wake feed jake original

Initial: Pirates of the Mississippi

Final: If I die before I wake / Feed Jake ...

Human status: pending; neither correctness nor support has been judged.

## 42. NQ / heldout / 4835361556991861440

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `96545134df70b2de7221e304b8e9e9bdf96ef171142e2e53ad627e44537214b0`.

Question: who was the first to declare the mutiny against the british

Initial: Benjamin Disraeli

Final: 10 May 1857

Human status: pending; neither correctness nor support has been judged.

## 43. NQ / heldout / 8222114783349850286

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `a7998a0f3eaa75c509d0d8dc6746d88d3c21367f201b3afaf7c87fd5524882c7`.

Question: who sets the requirements of a single audit

Initial: OMB

Final: OMB Circular A-133 and its Compliance Supplement

Human status: pending; neither correctness nor support has been judged.


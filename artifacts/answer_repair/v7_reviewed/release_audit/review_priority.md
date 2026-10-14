# E 新输出人工审核顺序

排序仅依据冻结指标和是否改答案，不代表人工结论。541 条结果，536 个唯一 dataset/sample_id；各批分别保留。

优先审核前 43 条改动（18 条指标退步、13 条指标改善、12 条其他改动）；其余 498 条保留答案的结果仍待完整证据支持审核。

人工需要对同一条的初始答案与最终答案分别判定：正确性、证据支持，共四项；记录审核人和理由。
请把确认写入 review_priority.jsonl 的 pair_review，完成时设 status=completed；四个布尔字段不能为空。
原始 pending_semantic_pairs.jsonl 保留为不可改写的审计底稿。不要用冻结参考自动填人工字段。

已有正确性判断也要按同一口径复查。引用字面来自原文，只能证明来源合法；返回原答案不代表答案已获证实。

## 1. FEVER / heldout / 126292

优先原因：frozen_metric_regression；结果指纹：`a5d2381304c29ae136088cd7400d25ec931c3bb0ffdc00db9af35714bed8ea38`。

问题：Camden, New Jersey is a county.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 2. FEVER / heldout / 14448

优先原因：frozen_metric_regression；结果指纹：`7117a5f59ba0b48be66943865b19b186519ed5b4738db10f26e310b526405308`。

问题：Sexual slavery is a reason for human trafficking.

初始：SUPPORTS

最终：REFUTES

人工状态：pending，正确性与支持性均未判定。

## 3. FEVER / heldout / 164986

优先原因：frozen_metric_regression；结果指纹：`371a0da97836b0376460ab136efcaa22e9b53f383773f2810d177d410782c429`。

问题：Polar bears are classified as endangered.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 4. FEVER / heldout / 49192

优先原因：frozen_metric_regression；结果指纹：`dc2111f0dbe8384b2021beac7390534955d36ad256c95d7412e83ce4ab9b8738`。

问题：The Armenian Genocide rarely was the extermination of Armenians who were mostly Ottoman citizens.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 5. NQ / heldout / -1145193137828819774

优先原因：frozen_metric_regression；结果指纹：`1554248420ddb3cf979d9d5e4942d6bac46e92553dee20d6e6bb01da05c27082`。

问题：who plays the grandmother in game of thrones

初始：Rigg

最终：Lady Olenna Tyrell

人工状态：pending，正确性与支持性均未判定。

## 6. NQ / dev50 / -1450346568093066821

优先原因：frozen_metric_regression；结果指纹：`52fe64f765b5088f9c2bfb01faf35c8422c6fda77633ca5043b15e61d16df802`。

问题：where was because i could not stop for death published

初始：Poems: Series 1

最终：in Poems : Series 1 in 1890

人工状态：pending，正确性与支持性均未判定。

## 7. NQ / heldout / -2324904500814145665

优先原因：frozen_metric_regression；结果指纹：`3e78faa1cb9c4bab0137eabfe517d9d31769028a9d9b4fc4dc2a2ea029b306da`。

问题：when was south african youth day first celebrated

初始：1976

最终：16 June

人工状态：pending，正确性与支持性均未判定。

## 8. NQ / heldout / -3719623571078658627

优先原因：frozen_metric_regression；结果指纹：`38d03d14f364a87aa9e3edb8e4dc69242555082eaaa90f977fa44a37df6062b4`。

问题：what does rt mean on a dodge car

初始：Road/Track

最终：Road / Track

人工状态：pending，正确性与支持性均未判定。

## 9. NQ / heldout / -5788046732019278353

优先原因：frozen_metric_regression；结果指纹：`02f3a4dcafe2bb11661aa7a01f0ec519d8d703034b8623fa648ef4e7b1d0a0b9`。

问题：who talks for belle in beauty and the beast

初始：Paige O'Hara

最终：American actress and singer Paige O'Hara

人工状态：pending，正确性与支持性均未判定。

## 10. NQ / heldout / -7524627393810362735

优先原因：frozen_metric_regression；结果指纹：`5e01ce4a43e8089732fd49a8a72a8afe673cca1ee36b4af6c17f1f5b80fcf178`。

问题：when did men's curling start in the olympics

初始：1998

最终：1924

人工状态：pending，正确性与支持性均未判定。

## 11. NQ / heldout / -7583855073136179893

优先原因：frozen_metric_regression；结果指纹：`b34a797bad38c387c8147db90aa5036aeb8a4cb22633a299687fd46c11103775`。

问题：when did david akers kick the 63 yard field goal

初始：September 9, 2012

最终：On September 9 , 2012

人工状态：pending，正确性与支持性均未判定。

## 12. NQ / heldout / -7778233237890403173

优先原因：frozen_metric_regression；结果指纹：`085c635910988e07485dded152641c4180e5a9235edf5cd191bc37ad2d2acb8d`。

问题：where did they film the show the crossing

初始：British Columbia, Canada

最终：coastal areas of British Columbia and in the city of Vancouver , British Columbia

人工状态：pending，正确性与支持性均未判定。

## 13. NQ / heldout / -8421690753817748624

优先原因：frozen_metric_regression；结果指纹：`ce9dd8cc3092df8fef6636d7e156f548809cd4d5704dbde53bf834b291eb7e02`。

问题：where is the heart in the human body

初始：between the lungs, in the middle compartment of the chest

最终：in the middle mediastinum , at the level of thoracic vertebrae T5 - T8

人工状态：pending，正确性与支持性均未判定。

## 14. NQ / dev50 / -8825841420882619284

优先原因：frozen_metric_regression；结果指纹：`84bc326d9d47d374d9e90480f3ce3c5b17ddfe6eb6df427d75286ac278e43d06`。

问题：where's the tv show the crossing filmed

初始：British Columbia, Canada

最终：coastal areas of British Columbia and in the city of Vancouver , British Columbia

人工状态：pending，正确性与支持性均未判定。

## 15. NQ / heldout / 2776741854318333179

优先原因：frozen_metric_regression；结果指纹：`a71af8b2cbbcf093e7b1cd26f34b1b5efcb52435edb93bac9f9a61e0ee131652`。

问题：who sings only love can break your heart

初始：Neil Young

最终：Saint Etienne

人工状态：pending，正确性与支持性均未判定。

## 16. NQ / heldout / 3181001754120908131

优先原因：frozen_metric_regression；结果指纹：`547c45c889ed6ae1a9230a954d4dad557eb58ece38fb1e2d40e952098aed3360`。

问题：india's first arctic research station set up in 2008 what is the name

初始：Himadri Station

最终：Himadri

人工状态：pending，正确性与支持性均未判定。

## 17. NQ / heldout / 3630714342371225513

优先原因：frozen_metric_regression；结果指纹：`b0d5229adb1c643997c79d6d7d78c87f8b6f6859b07278f3761185b218cfd746`。

问题：where did the term spirits for alcohol come from

初始：Middle Eastern alchemy

最终：spirit of the original material

人工状态：pending，正确性与支持性均未判定。

## 18. NQ / dev50 / 5898512601992001509

优先原因：frozen_metric_regression；结果指纹：`f5789ea704a394db4a13da24cec202eba72852209e725d06b5b7f1c04185e732`。

问题：who did the united states fight in the war of 1812

初始：the United Kingdom

最终：the United States , the United Kingdom , and their respective allies

人工状态：pending，正确性与支持性均未判定。

## 19. FEVER / reviewed / 146639

优先原因：frozen_metric_gain；结果指纹：`0dfc1acb9cd422a01e11b8143d8d2153e7da05916be079316ce44d44008c7aa6`。

问题：In 2003, Eva Green debuted in film.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 20. FEVER / reviewed / 156493

优先原因：frozen_metric_gain；结果指纹：`df0ab6e1c950ab263dae533ef28aa0e09007e81ee7be6a2b9d86a3e770e060b5`。

问题：Daggering is not a type of dance.

初始：SUPPORTS

最终：REFUTES

人工状态：pending，正确性与支持性均未判定。

## 21. FEVER / reviewed / 161905

优先原因：frozen_metric_gain；结果指纹：`409647d9e8a19d711888554f4bd2fe66d8ae1b7505df8bd7ca461b5343b74067`。

问题：In 1974 Alan Shepard retired from the US Navy.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 22. FEVER / reviewed / 203615

优先原因：frozen_metric_gain；结果指纹：`0893b1cc3d8e8c9fa06f9190c30d112d51876a161ca5ce399429c0d063d1315b`。

问题：The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 23. FEVER / reviewed / 71180

优先原因：frozen_metric_gain；结果指纹：`0edf020f516ecea35bff6195aca9f27577f77a6278585434c71440f64502193f`。

问题：Babur was a Mughal emperor.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 24. FEVER / reviewed / 72578

优先原因：frozen_metric_gain；结果指纹：`9c3d56dadec77b7b00683e29b78fb737f681a64d9a2c1e3ddd669a9e9a8179c2`。

问题：The Greek language has speakers in Albania.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 25. FEVER / heldout / 104922

优先原因：frozen_metric_gain；结果指纹：`539fda5c8077c7ea14d85a03ef8d8819cea48442f9bc3a881f95aa4fc49a0939`。

问题：Oscar Isaac did not act as the supervillain Apocalypse.

初始：SUPPORTS

最终：REFUTES

人工状态：pending，正确性与支持性均未判定。

## 26. FEVER / dev50 / 203615

优先原因：frozen_metric_gain；结果指纹：`d013f2d70808e98175a574ddceb98748dedb7b83fb2166e4819fb1db877efed2`。

问题：The 1974 crime film The Sugarland Express was co-written by Steven Spielberg.

初始：REFUTES

最终：SUPPORTS

人工状态：pending，正确性与支持性均未判定。

## 27. NQ / reviewed / -2905696611778901658

优先原因：frozen_metric_gain；结果指纹：`aee1d7bb6dadd74e42d35fe5ff862a70c459bf40f23813218c09e2c25f4c5b00`。

问题：who sang oh annie i not your daddy

初始：Annie, I'm Not Your Daddy

最终：Kid Creole and the Coconuts

人工状态：pending，正确性与支持性均未判定。

## 28. NQ / dev50 / -2905696611778901658

优先原因：frozen_metric_gain；结果指纹：`897598341ed2a0b95eb38256c00c423d0eeea337bb2a4430f27edb003b7da827`。

问题：who sang oh annie i not your daddy

初始：Annie, I'm Not Your Daddy

最终：Kid Creole and the Coconuts

人工状态：pending，正确性与支持性均未判定。

## 29. NQ / dev50 / 448742752032961281

优先原因：frozen_metric_gain；结果指纹：`a89341acc556f916187219bb254502dcbe4820670bd84f0a9529f0d63eda24f3`。

问题：why god save the queen and not king

初始：gender of the reigning monarch

最终：depending on the gender of the reigning monarch

人工状态：pending，正确性与支持性均未判定。

## 30. NQ / heldout / 5320268832129970522

优先原因：frozen_metric_gain；结果指纹：`08a28b85e9c856c5bf3bdf9b6f5568f4749b8b7e6f23084a1a6d22c9e9d057ae`。

问题：what is the number of basic units in the international system of units

初始：7

最终：seven

人工状态：pending，正确性与支持性均未判定。

## 31. NQ / heldout / 6453011233363353306

优先原因：frozen_metric_gain；结果指纹：`8cca4def02b98ef625208511e3d4f6654fe0392d32653cf046e6b4e669b326c0`。

问题：the heart muscle is stimulated to contract by electrical impulses which are generated where

初始：SA node

最终：the sinoatrial node

人工状态：pending，正确性与支持性均未判定。

## 32. NQ / reviewed / 2326758035558088175

优先原因：changed_without_frozen_correctness_gain；结果指纹：`999e11bf23b2e68c5be1d74dcb752981ffa79c9e47a534a41bd38716b715c538`。

问题：what is the meaning of to have a bias

初始：prejudice in favor of or against one thing, person, or group compared with another

最终：prejudice in favour of or against one thing , person , or group compared with another

人工状态：pending，正确性与支持性均未判定。

## 33. NQ / heldout / -19630582826733372

优先原因：changed_without_frozen_correctness_gain；结果指纹：`813ad447fb515d44a9dadb1b78aae0094a342d7fdcc307e651af43843fb2f732`。

问题：where is the taurus mountains located on a map

初始：southern Turkey

最终：in southern Turkey

人工状态：pending，正确性与支持性均未判定。

## 34. NQ / dev50 / -2743701347497238377

优先原因：changed_without_frozen_correctness_gain；结果指纹：`4e4e82e4194ed8da2c338dcd07eab634840ba50dbcea983182cf8b0e431237d2`。

问题：how oxygenated blood returns to the heart from the lungs

初始：pulmonary veins

最终：through pulmonary veins

人工状态：pending，正确性与支持性均未判定。

## 35. NQ / heldout / -4170119894384300076

优先原因：changed_without_frozen_correctness_gain；结果指纹：`e898ddffd16d456bab3b8dfe5a77dc4d97d329de4e0db2330bc90b8dfe4269df`。

问题：who do you play as in dragon age origins

初始：the player-controlled character

最终：the player - controlled character

人工状态：pending，正确性与支持性均未判定。

## 36. NQ / heldout / -5248648953323441399

优先原因：changed_without_frozen_correctness_gain；结果指纹：`92f9364893d6588bebf80e158cfd71ed3f0f2332e3810e6bdfb9d6db2aa511f4`。

问题：when did god save the queen became the national anthem

初始：1974

最终：19 April 1984

人工状态：pending，正确性与支持性均未判定。

## 37. NQ / dev50 / -544805225908484128

优先原因：changed_without_frozen_correctness_gain；结果指纹：`62ab2e4671bdb9645f78d17bbc74fc7eabaa7f333dec9790c23145ef5b404eee`。

问题：is greenland part of europe or north america

初始：North America

最终：an autonomous constituent country within the Kingdom of Denmark

人工状态：pending，正确性与支持性均未判定。

## 38. NQ / heldout / -6081006991734955423

优先原因：changed_without_frozen_correctness_gain；结果指纹：`d939893a3041dec5a75d8d323ff9d398b9bfb2b1ce1643415b0bc0b3ae002e93`。

问题：where did the battle of bonhomme richard take place

初始：Flamborough Head, Yorkshire

最终：off Flamborough Head , Yorkshire

人工状态：pending，正确性与支持性均未判定。

## 39. NQ / heldout / -8299469566867123031

优先原因：changed_without_frozen_correctness_gain；结果指纹：`32e760d2b0cc8ea222f5c31dc9d9cd6b1ffb63257bf39e52c7a6720bc7d4d0fb`。

问题：what is the name of india pakistan border

初始：International Border (IB)

最终：India and Pakistan Border , known locally as the International Border ( IB )

人工状态：pending，正确性与支持性均未判定。

## 40. NQ / dev50 / 1012541351190124000

优先原因：changed_without_frozen_correctness_gain；结果指纹：`94fa5c2376fcc1859e84547440ee04082c82725131a2cbefe5e8ddec08bf23f4`。

问题：what is the new touchback rule in the nfl

初始：touchback on the 25-yard line

最终：the 25 - yard line

人工状态：pending，正确性与支持性均未判定。

## 41. NQ / dev50 / 2178109984531253647

优先原因：changed_without_frozen_correctness_gain；结果指纹：`a25beffc9bc37c6e5ebaaabb5525bb127de24888556ac695a99cc0e6c787712f`。

问题：if i die before i wake feed jake original

初始：Pirates of the Mississippi

最终：If I die before I wake / Feed Jake ...

人工状态：pending，正确性与支持性均未判定。

## 42. NQ / heldout / 4835361556991861440

优先原因：changed_without_frozen_correctness_gain；结果指纹：`96545134df70b2de7221e304b8e9e9bdf96ef171142e2e53ad627e44537214b0`。

问题：who was the first to declare the mutiny against the british

初始：Benjamin Disraeli

最终：10 May 1857

人工状态：pending，正确性与支持性均未判定。

## 43. NQ / heldout / 8222114783349850286

优先原因：changed_without_frozen_correctness_gain；结果指纹：`a7998a0f3eaa75c509d0d8dc6746d88d3c21367f201b3afaf7c87fd5524882c7`。

问题：who sets the requirements of a single audit

初始：OMB

最终：OMB Circular A-133 and its Compliance Supplement

人工状态：pending，正确性与支持性均未判定。


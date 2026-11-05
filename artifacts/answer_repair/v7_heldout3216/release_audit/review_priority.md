# E new-output human review order

Ordering is based only on frozen metrics and whether the answer changed, and does not represent a human conclusion. 300 results, 300 unique dataset/sample_id; each batch is kept separately.

Review the first 52 changes first (9 metric regressions, 13 metric improvements, 30 other changes); the remaining 248 results that kept the answer still await full evidence-support review.

For the same item, the human reviewer must judge the initial answer and the final answer separately on correctness and evidence support, four items in total; record the reviewer and the rationale.
Write confirmations into pair_review in review_priority.jsonl, and set status=completed when done; the four boolean fields must not be empty.
The original pending_semantic_pairs.jsonl is kept as an immutable audit base record. Do not use the frozen references to auto-fill human fields.

Existing correctness judgments must also be rechecked under the same criteria. Citations taken verbatim from the source text only prove the source is legitimate; returning the original answer does not mean the answer has been verified.

## 1. NQ / heldout3216 / -3513582807303361660

Priority reason: frozen_metric_regression; result fingerprint: `c5efab2fc969da99432ac80aad01a66c22115b068163e71eece95549beaf1a98`.

Question: what was the full name of the titanic

Initial: RMS Titanic

Final: Titanic

Human status: pending; neither correctness nor support has been judged.

## 2. NQ / heldout3216 / -7958414179887081456

Priority reason: frozen_metric_regression; result fingerprint: `1feafc2b4fe65ea6c04978b9b8da856d804e0d2f85f201c5dc56be853f0cac0a`.

Question: how does a solid state fuel pump work

Initial: via pulse-width modulation of the pump voltage

Final: via pulse - width modulation of the pump voltage

Human status: pending; neither correctness nor support has been judged.

## 3. NQ / heldout3216 / -86061704964093180

Priority reason: frozen_metric_regression; result fingerprint: `83b4e073d2a054b207f4c7ac89c6a54edf8935ebdc75e48d43515ab6ab97b760`.

Question: when was the suite life of zack and cody made

Initial: 2005

Final: March 18 , 2005

Human status: pending; neither correctness nor support has been judged.

## 4. NQ / heldout3216 / 2963242422999718442

Priority reason: frozen_metric_regression; result fingerprint: `745f3b1144e1da52e085a7ecea148136c1a4cf6251346e71ec1342e410cfd87f`.

Question: during which season does cape town receive rainfall

Initial: winter

Final: more in the wintertime

Human status: pending; neither correctness nor support has been judged.

## 5. NQ / heldout3216 / 5828164811839577668

Priority reason: frozen_metric_regression; result fingerprint: `8fb876f964e15dc29b0751983d99055d81e2bf50fc4a1d8f76aa32acc270665a`.

Question: the temperature of gas is proportional to the

Initial: volume

Final: the volume of a gas

Human status: pending; neither correctness nor support has been judged.

## 6. NQ / heldout3216 / 6466066334172447213

Priority reason: frozen_metric_regression; result fingerprint: `1373cdb64dabf272caf2eb1975f14d528c31ada12f0304459fadfe6c9f0dd7df`.

Question: when did the us not go to the olympics

Initial: 1980

Final: 1980 summer Olympics

Human status: pending; neither correctness nor support has been judged.

## 7. NQ / heldout3216 / 8324017764283655875

Priority reason: frozen_metric_regression; result fingerprint: `3a179c1d55bd16debaf620d989ef49a8f4af45550185fd780d19b8bac512b616`.

Question: what is the oldest street in the philippines

Initial: Colon Street

Final: Elfreth 's Alley

Human status: pending; neither correctness nor support has been judged.

## 8. NQ / heldout3216 / 8911390210346559626

Priority reason: frozen_metric_regression; result fingerprint: `18fce57395cdeb227b733301e1a352b28b1fcf960ddbda40f6bd4397472f46df`.

Question: who plays the dad in malcom in the middle

Initial: Bryan Cranston

Final: Hal ( Bryan Cranston )

Human status: pending; neither correctness nor support has been judged.

## 9. NQ / heldout3216 / 95636137517606056

Priority reason: frozen_metric_regression; result fingerprint: `ec0ad10057f323c70dcc17c9984096ac04cb8209b9bdda671f9134e57d2afd46`.

Question: tallest building in the world of all time

Initial: Burj Khalifa

Final: Burj Khalifa in Dubai , United Arab Emirates

Human status: pending; neither correctness nor support has been judged.

## 10. NQ / heldout3216 / -1626541047372980132

Priority reason: frozen_metric_gain; result fingerprint: `e8f7c7af7350c17a5193745942f379e0b3b5b97d406dd0ff32918e6d99047283`.

Question: when did the red guards start and end

Initial: 1966 to 1976

Final: 1966 and 1967

Human status: pending; neither correctness nor support has been judged.

## 11. NQ / heldout3216 / -5702657710806575402

Priority reason: frozen_metric_gain; result fingerprint: `2a7fddf2ffd0261f0e495eb11cc055a20ea1e78386ce9bd013eae91e6bfb9aed`.

Question: who sang nice day for a white wedding

Initial: In This Moment

Final: Billy Idol

Human status: pending; neither correctness nor support has been judged.

## 12. NQ / heldout3216 / -6655692357261534837

Priority reason: frozen_metric_gain; result fingerprint: `ca0e2d918d4a6dbb7975ff56e29e88d5df931bd0f2a2ea082026dd71cd100abe`.

Question: when does season 6 of the next step start

Initial: March 4, 2014

Final: 2018

Human status: pending; neither correctness nor support has been judged.

## 13. NQ / heldout3216 / -8590084884830548313

Priority reason: frozen_metric_gain; result fingerprint: `72f96703e3e3b4ffc0c0e35eb5933f75af0f935deaee53cdc06f85fa64852a75`.

Question: when does season 13 of america's got talent premiere

Initial: January 15, 2014

Final: May 29 , 2018

Human status: pending; neither correctness nor support has been judged.

## 14. NQ / heldout3216 / 1747539378277902434

Priority reason: frozen_metric_gain; result fingerprint: `8888d246b58f1e15b0b9bd8c3dff63d309b98761ace3df967b7fd9d81d6dffa3`.

Question: who sang smoke gets in your eyes first

Initial: Paul Whiteman and his Orchestra

Final: Tamara Drasin

Human status: pending; neither correctness nor support has been judged.

## 15. NQ / heldout3216 / 2495228512593253768

Priority reason: frozen_metric_gain; result fingerprint: `37e45016a1152c72ab5bfc9d22a98f662465d434b5b88fa7897a845dbaa5588a`.

Question: when did we decide to leave the eu

Initial: 22 February 2016

Final: 23 June 2016

Human status: pending; neither correctness nor support has been judged.

## 16. NQ / heldout3216 / 275718753854271176

Priority reason: frozen_metric_gain; result fingerprint: `2ffaaa5aa799de4cb8741806bdfdeaaca6d31a92a05291faf9e97f9ec58210d4`.

Question: who has been designed as the new finance secretary of india

Initial: Not specified

Final: Hasmukh Adhia

Human status: pending; neither correctness nor support has been judged.

## 17. NQ / heldout3216 / 3786291662182153375

Priority reason: frozen_metric_gain; result fingerprint: `825c9c99628947cced5de8e5241b8ec1e8361ae6bb427b4bf08c5aca36af8a2e`.

Question: when did the crucifix become the symbol of christianity

Initial: 6th century AD

Final: 4th century

Human status: pending; neither correctness nor support has been judged.

## 18. NQ / heldout3216 / 3886074985605209321

Priority reason: frozen_metric_gain; result fingerprint: `5646ac9f603b00f00841a9b766067a305b3a1147f6afeb46e62c948fcb48159a`.

Question: who sings find out who your friends are with tracy lawrence

Initial: Tracy Lawrence

Final: Tim McGraw and Kenny Chesney

Human status: pending; neither correctness nor support has been judged.

## 19. NQ / heldout3216 / 5328438809332726480

Priority reason: frozen_metric_gain; result fingerprint: `b4716deadcb6df604bc999d914a7c3a654605c45f925767f6dbdee75b77c21a8`.

Question: when does planet of the apes come out 2017

Initial: October 10, 2017

Final: July 14 , 2017

Human status: pending; neither correctness nor support has been judged.

## 20. NQ / heldout3216 / 6339911987422116455

Priority reason: frozen_metric_gain; result fingerprint: `e50944eed60b455cea18add830237bbf6ce80fc974a802c643bbebe090f9f4f9`.

Question: when was the last time astros was in the world series

Initial: 2005

Final: 2017

Human status: pending; neither correctness nor support has been judged.

## 21. NQ / heldout3216 / 7217222058435937287

Priority reason: frozen_metric_gain; result fingerprint: `a535091c85848067359118f0fe5908ddea49b471dc09edb55e7c2e201ee3a312`.

Question: where was the world economic forum held this year

Initial: Information not provided in the context.

Final: Davos , a mountain resort in Graubünden , in the eastern Alps region of Switzerland

Human status: pending; neither correctness nor support has been judged.

## 22. NQ / heldout3216 / 7273651277313451649

Priority reason: frozen_metric_gain; result fingerprint: `2287f72b8a1a52166efcaaf863cff51f6c362209b5d21fd167ff8070c8425977`.

Question: when will fairy tail final season be released

Initial: March 8, 2018

Final: 2018

Human status: pending; neither correctness nor support has been judged.

## 23. NQ / heldout3216 / -1317942101497986450

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `b6746d2e55fb31a03ee98aad9ad3a05bdbddd4f24a0b2cbb74672961c238719b`.

Question: when is season 8 for game of thrones

Initial: 2018 or 2019

Final: July 2016

Human status: pending; neither correctness nor support has been judged.

## 24. NQ / heldout3216 / -1514464247537659362

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `3e2d0279719649f91d7b53ddf7c1d6d3830cf947a01a5c35ec2c0c573819c55d`.

Question: when does the sword art online movie take place

Initial: after the anime series' second season

Final: after the anime series ' second season , Sword Art Online II

Human status: pending; neither correctness nor support has been judged.

## 25. NQ / heldout3216 / -1910743846708408775

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `e60463a01d93059180405bbb40123a2280176a7f02c84a6bacaf1fd71d50a9d0`.

Question: when was the taming ofthe shrew first performed

Initial: not specified

Final: between 1590 and 1592

Human status: pending; neither correctness nor support has been judged.

## 26. NQ / heldout3216 / -3276477249293599145

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `c69d442b6890a08a5528f2dab06f6802571462eff7d39081319425e3aa601a88`.

Question: when does law of diminishing returns set in

Initial: when adding more of a factor of production, while holding all others constant

Final: at some point

Human status: pending; neither correctness nor support has been judged.

## 27. NQ / heldout3216 / -3828702383571778476

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `0c8710326635302703e32adb460e56570de9247c05da28023bc87c1c453e9a33`.

Question: what is the meaning of the name von

Initial: a noble patrilineality or of/from

Final: a nobiliary particle indicating a noble patrilineality or as a simple preposition

Human status: pending; neither correctness nor support has been judged.

## 28. NQ / heldout3216 / -3984706621862511717

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `b8288466ce20f0e9652745ba0f891ae19ba11c093e67987fe999d07466be17e2`.

Question: where does cortisol come from in the body

Initial: zona fasciculata of the adrenal cortex

Final: the adrenal gland in the zona fasciculata

Human status: pending; neither correctness nor support has been judged.

## 29. NQ / heldout3216 / -4546405080702371934

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `814c6cb842873de901bec72a75855b79c0137fc58cad02d6fbe7b95beb7aec45`.

Question: when was the term prime minister first used

Initial: 18th century

Final: 18th century in the United Kingdom

Human status: pending; neither correctness nor support has been judged.

## 30. NQ / heldout3216 / -4822862932182551070

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `5ab91552d217e3b1ab7788e23ef50ed3d839a42c105ad53a39ed250231efeb44`.

Question: another name for a hairpin bend crossword clue

Initial: bobby pin

Final: hairpin bend , hairpin corner

Human status: pending; neither correctness nor support has been judged.

## 31. NQ / heldout3216 / -5327690920099764097

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `1a12bf4cbc0a6da4c18d5df84a42216cd52f78d9694bbc335b49763e6e764c90`.

Question: where does the name elevator pitch come from

Initial: approximately thirty seconds to two minutes

Final: it should be possible to deliver the summary in the time span of an elevator ride

Human status: pending; neither correctness nor support has been judged.

## 32. NQ / heldout3216 / -5551039786517755042

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `428225869102bde44a826fa7db967ee7df0d80debd25e91a198a48806507a854`.

Question: who sings there's a hole in my bucket

Initial: children

Final: a children 's song

Human status: pending; neither correctness nor support has been judged.

## 33. NQ / heldout3216 / -5623229691925051533

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `d002951941e1b278e81c32297ff0051041ce54adc308d7f20bb37119c50b69ca`.

Question: when was the south asian association for regional co-operation (saarc) formed

Initial: 1985

Final: 8 December 1985

Human status: pending; neither correctness nor support has been judged.

## 34. NQ / heldout3216 / -5765931546365345480

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `14ec49b0e64110026908d3e197f6239207ed48a4213fcd65f284b722be511a1e`.

Question: where does kevins family go in home alone 2

Initial: New York

Final: New York City

Human status: pending; neither correctness nor support has been judged.

## 35. NQ / heldout3216 / -6751220433242447969

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `d13c56275dfaf5118f5c0904dfb7fc03b4337f694975c44e7f80456d9c551208`.

Question: how many seasons of the bastard executioner are there

Initial: 4

Final: one season

Human status: pending; neither correctness nor support has been judged.

## 36. NQ / heldout3216 / -7721509949632484180

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `7bf2290699f463a78a340c3fc4c6a63053bcd9d91c42689673489587f97ef34f`.

Question: when was the first driver's license required

Initial: August 1, 1910

Final: On August 1 , 1910

Human status: pending; neither correctness nor support has been judged.

## 37. NQ / heldout3216 / -7900537827363346667

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `eb6f55fa284baae4a8cfe4672230ae12525fb76b62be832523fc80aca9866fc9`.

Question: where would a subcutaneous injection be made in the skin

Initial: under the skin

Final: under the skin , into the fat layer

Human status: pending; neither correctness nor support has been judged.

## 38. NQ / heldout3216 / -8006002028754190814

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `44a602d0bd9856f863caf9807583d3a04b30f484424551083d6756954c9634b8`.

Question: who had the most governmental power under the articles of confederation

Initial: Congress of the Confederation

Final: the central government 's power was kept quite limited

Human status: pending; neither correctness nor support has been judged.

## 39. NQ / heldout3216 / -8696133314731800801

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `81a1ed629bb54e40930334d1411545ca93f4bba42db691164b14db1d3c41ab2c`.

Question: when was the chain first used for f1

Initial: Never

Final: 1973 Canadian Grand Prix

Human status: pending; neither correctness nor support has been judged.

## 40. NQ / heldout3216 / -8767841388599751671

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `ec464cbee5778d2c18a6f5e41b0129a51135b00d5d799a92ed082ca907f90030`.

Question: who coined the phrase let them eat cake

Initial: Alphonse Karr

Final: a great princess

Human status: pending; neither correctness nor support has been judged.

## 41. NQ / heldout3216 / 2304123337731564294

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `e4ee8282b0456beb880b3c58751508c60880395b79630d25120cb91638375ef4`.

Question: what do the economic systems of the uk germany and russia have in common

Initial: None

Final: $2.3 million

Human status: pending; neither correctness nor support has been judged.

## 42. NQ / heldout3216 / 2401915830495655749

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `1a930897e3dd0514094ff4b791836a4026ed8b91036d7e64a2f24064ce237552`.

Question: what percentage of the us population lives below the poverty line

Initial: 15.1%

Final: 13.5 %

Human status: pending; neither correctness nor support has been judged.

## 43. NQ / heldout3216 / 2918354452051217435

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `ccfdfe74f435741ba579c239913e5f3d537a48a55464001fe84c17c0fc623a3c`.

Question: where was the first high level language operating system created

Initial: Germany

Final: Plankalkül , created by Konrad Zuse between 1942 and 1945

Human status: pending; neither correctness nor support has been judged.

## 44. NQ / heldout3216 / 3630480194417653112

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `8c37037b5ac2944f10e5a4695df9ca3086e0bb7e2376618e5fc08cc513b1dde5`.

Question: what is henry james the real thing about

Initial: the reality-illusion dichotomy

Final: The Real Thing and Other Stories published by Macmillan .

Human status: pending; neither correctness nor support has been judged.

## 45. NQ / heldout3216 / 4140332518759407550

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `56948f73f4c92e472c21f7fc1532c26c518684bc522fa814437ab11eae6d6f97`.

Question: where did spock's hand sign come from

Initial: Vulcan culture

Final: the Priestly Blessing performed by Jewish Kohanim

Human status: pending; neither correctness nor support has been judged.

## 46. NQ / heldout3216 / 5482109430613356344

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `620d8b50c6206a5d89044417ce839f82deb8cb3fdb48a8506ab30cf40d08927b`.

Question: where did the rulers of the qing dynasty originate

Initial: Jurchen

Final: a sedentary farming people known as the Jurchen

Human status: pending; neither correctness nor support has been judged.

## 47. NQ / heldout3216 / 6276969298797177537

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `cc17b0c4215da066d4393e9492b92b142f61ff4efb959bc8df35ebe5faf0cd52`.

Question: what is the main objective of criminal law

Initial: punishment

Final: retribution

Human status: pending; neither correctness nor support has been judged.

## 48. NQ / heldout3216 / 6419695748325791865

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `c5e2af902cac4304732d9899be918a30a7996d37bdf4075fb589ad7f816da516`.

Question: who plays david in alvin and the chipmunks

Initial: Ross Bagdasarian Jr.

Final: Ross Bagdasarian Sr.

Human status: pending; neither correctness nor support has been judged.

## 49. NQ / heldout3216 / 7206245989456268244

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `5f85f6fe56999bcace3551f7928547e88fcc1441f62be452510a4de165190ef0`.

Question: who wrote it's gonna take a lot of love

Initial: It's not "It's Gonna Take a Lot of Love", but "It's Gonna Take a Miracle". Written by Teddy Randazzo, Bobby Weinstein, and Lou Stallman.

Final: Teddy Randazzo , Bobby Weinstein , and Lou Stallman

Human status: pending; neither correctness nor support has been judged.

## 50. NQ / heldout3216 / 7520402739582397133

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `bbed42c75fe9c8ef7f75d6286e98d018a4dbb709f01e026606e2d5975bffbfd1`.

Question: when was the first book made into a movie

Initial: 2001

Final: 16 November 2001

Human status: pending; neither correctness nor support has been judged.

## 51. NQ / heldout3216 / 795441882608131318

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `29fcb889bb16a375858e4f9accb10e9ac89e9b982476250492fe3d174f176544`.

Question: where are the four disney theme parks located

Initial: Walt Disney World Resort, Disneyland Paris, Tokyo Disneyland, and none specified for Europe

Final: Lake Buena Vista , Florida

Human status: pending; neither correctness nor support has been judged.

## 52. NQ / heldout3216 / 8803717720023296980

Priority reason: changed_without_frozen_correctness_gain; result fingerprint: `d2663dfee1f0da8f0d562928a07f399846fcfbe20b86dab3ae784304fb3c62b2`.

Question: where does the sciatic nerve run in the foot

Initial: entire lower leg ( except for its inner side )

Final: supplies sensation to the skin of the foot

Human status: pending; neither correctness nor support has been judged.


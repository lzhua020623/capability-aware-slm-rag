# E 新输出人工审核顺序

排序仅依据冻结指标和是否改答案，不代表人工结论。300 条结果，300 个唯一 dataset/sample_id；各批分别保留。

优先审核前 52 条改动（9 条指标退步、13 条指标改善、30 条其他改动）；其余 248 条保留答案的结果仍待完整证据支持审核。

人工需要对同一条的初始答案与最终答案分别判定：正确性、证据支持，共四项；记录审核人和理由。
请把确认写入 review_priority.jsonl 的 pair_review，完成时设 status=completed；四个布尔字段不能为空。
原始 pending_semantic_pairs.jsonl 保留为不可改写的审计底稿。不要用冻结参考自动填人工字段。

已有正确性判断也要按同一口径复查。引用字面来自原文，只能证明来源合法；返回原答案不代表答案已获证实。

## 1. NQ / heldout3216 / -3513582807303361660

优先原因：frozen_metric_regression；结果指纹：`c5efab2fc969da99432ac80aad01a66c22115b068163e71eece95549beaf1a98`。

问题：what was the full name of the titanic

初始：RMS Titanic

最终：Titanic

人工状态：pending，正确性与支持性均未判定。

## 2. NQ / heldout3216 / -7958414179887081456

优先原因：frozen_metric_regression；结果指纹：`1feafc2b4fe65ea6c04978b9b8da856d804e0d2f85f201c5dc56be853f0cac0a`。

问题：how does a solid state fuel pump work

初始：via pulse-width modulation of the pump voltage

最终：via pulse - width modulation of the pump voltage

人工状态：pending，正确性与支持性均未判定。

## 3. NQ / heldout3216 / -86061704964093180

优先原因：frozen_metric_regression；结果指纹：`83b4e073d2a054b207f4c7ac89c6a54edf8935ebdc75e48d43515ab6ab97b760`。

问题：when was the suite life of zack and cody made

初始：2005

最终：March 18 , 2005

人工状态：pending，正确性与支持性均未判定。

## 4. NQ / heldout3216 / 2963242422999718442

优先原因：frozen_metric_regression；结果指纹：`745f3b1144e1da52e085a7ecea148136c1a4cf6251346e71ec1342e410cfd87f`。

问题：during which season does cape town receive rainfall

初始：winter

最终：more in the wintertime

人工状态：pending，正确性与支持性均未判定。

## 5. NQ / heldout3216 / 5828164811839577668

优先原因：frozen_metric_regression；结果指纹：`8fb876f964e15dc29b0751983d99055d81e2bf50fc4a1d8f76aa32acc270665a`。

问题：the temperature of gas is proportional to the

初始：volume

最终：the volume of a gas

人工状态：pending，正确性与支持性均未判定。

## 6. NQ / heldout3216 / 6466066334172447213

优先原因：frozen_metric_regression；结果指纹：`1373cdb64dabf272caf2eb1975f14d528c31ada12f0304459fadfe6c9f0dd7df`。

问题：when did the us not go to the olympics

初始：1980

最终：1980 summer Olympics

人工状态：pending，正确性与支持性均未判定。

## 7. NQ / heldout3216 / 8324017764283655875

优先原因：frozen_metric_regression；结果指纹：`3a179c1d55bd16debaf620d989ef49a8f4af45550185fd780d19b8bac512b616`。

问题：what is the oldest street in the philippines

初始：Colon Street

最终：Elfreth 's Alley

人工状态：pending，正确性与支持性均未判定。

## 8. NQ / heldout3216 / 8911390210346559626

优先原因：frozen_metric_regression；结果指纹：`18fce57395cdeb227b733301e1a352b28b1fcf960ddbda40f6bd4397472f46df`。

问题：who plays the dad in malcom in the middle

初始：Bryan Cranston

最终：Hal ( Bryan Cranston )

人工状态：pending，正确性与支持性均未判定。

## 9. NQ / heldout3216 / 95636137517606056

优先原因：frozen_metric_regression；结果指纹：`ec0ad10057f323c70dcc17c9984096ac04cb8209b9bdda671f9134e57d2afd46`。

问题：tallest building in the world of all time

初始：Burj Khalifa

最终：Burj Khalifa in Dubai , United Arab Emirates

人工状态：pending，正确性与支持性均未判定。

## 10. NQ / heldout3216 / -1626541047372980132

优先原因：frozen_metric_gain；结果指纹：`e8f7c7af7350c17a5193745942f379e0b3b5b97d406dd0ff32918e6d99047283`。

问题：when did the red guards start and end

初始：1966 to 1976

最终：1966 and 1967

人工状态：pending，正确性与支持性均未判定。

## 11. NQ / heldout3216 / -5702657710806575402

优先原因：frozen_metric_gain；结果指纹：`2a7fddf2ffd0261f0e495eb11cc055a20ea1e78386ce9bd013eae91e6bfb9aed`。

问题：who sang nice day for a white wedding

初始：In This Moment

最终：Billy Idol

人工状态：pending，正确性与支持性均未判定。

## 12. NQ / heldout3216 / -6655692357261534837

优先原因：frozen_metric_gain；结果指纹：`ca0e2d918d4a6dbb7975ff56e29e88d5df931bd0f2a2ea082026dd71cd100abe`。

问题：when does season 6 of the next step start

初始：March 4, 2014

最终：2018

人工状态：pending，正确性与支持性均未判定。

## 13. NQ / heldout3216 / -8590084884830548313

优先原因：frozen_metric_gain；结果指纹：`72f96703e3e3b4ffc0c0e35eb5933f75af0f935deaee53cdc06f85fa64852a75`。

问题：when does season 13 of america's got talent premiere

初始：January 15, 2014

最终：May 29 , 2018

人工状态：pending，正确性与支持性均未判定。

## 14. NQ / heldout3216 / 1747539378277902434

优先原因：frozen_metric_gain；结果指纹：`8888d246b58f1e15b0b9bd8c3dff63d309b98761ace3df967b7fd9d81d6dffa3`。

问题：who sang smoke gets in your eyes first

初始：Paul Whiteman and his Orchestra

最终：Tamara Drasin

人工状态：pending，正确性与支持性均未判定。

## 15. NQ / heldout3216 / 2495228512593253768

优先原因：frozen_metric_gain；结果指纹：`37e45016a1152c72ab5bfc9d22a98f662465d434b5b88fa7897a845dbaa5588a`。

问题：when did we decide to leave the eu

初始：22 February 2016

最终：23 June 2016

人工状态：pending，正确性与支持性均未判定。

## 16. NQ / heldout3216 / 275718753854271176

优先原因：frozen_metric_gain；结果指纹：`2ffaaa5aa799de4cb8741806bdfdeaaca6d31a92a05291faf9e97f9ec58210d4`。

问题：who has been designed as the new finance secretary of india

初始：Not specified

最终：Hasmukh Adhia

人工状态：pending，正确性与支持性均未判定。

## 17. NQ / heldout3216 / 3786291662182153375

优先原因：frozen_metric_gain；结果指纹：`825c9c99628947cced5de8e5241b8ec1e8361ae6bb427b4bf08c5aca36af8a2e`。

问题：when did the crucifix become the symbol of christianity

初始：6th century AD

最终：4th century

人工状态：pending，正确性与支持性均未判定。

## 18. NQ / heldout3216 / 3886074985605209321

优先原因：frozen_metric_gain；结果指纹：`5646ac9f603b00f00841a9b766067a305b3a1147f6afeb46e62c948fcb48159a`。

问题：who sings find out who your friends are with tracy lawrence

初始：Tracy Lawrence

最终：Tim McGraw and Kenny Chesney

人工状态：pending，正确性与支持性均未判定。

## 19. NQ / heldout3216 / 5328438809332726480

优先原因：frozen_metric_gain；结果指纹：`b4716deadcb6df604bc999d914a7c3a654605c45f925767f6dbdee75b77c21a8`。

问题：when does planet of the apes come out 2017

初始：October 10, 2017

最终：July 14 , 2017

人工状态：pending，正确性与支持性均未判定。

## 20. NQ / heldout3216 / 6339911987422116455

优先原因：frozen_metric_gain；结果指纹：`e50944eed60b455cea18add830237bbf6ce80fc974a802c643bbebe090f9f4f9`。

问题：when was the last time astros was in the world series

初始：2005

最终：2017

人工状态：pending，正确性与支持性均未判定。

## 21. NQ / heldout3216 / 7217222058435937287

优先原因：frozen_metric_gain；结果指纹：`a535091c85848067359118f0fe5908ddea49b471dc09edb55e7c2e201ee3a312`。

问题：where was the world economic forum held this year

初始：Information not provided in the context.

最终：Davos , a mountain resort in Graubünden , in the eastern Alps region of Switzerland

人工状态：pending，正确性与支持性均未判定。

## 22. NQ / heldout3216 / 7273651277313451649

优先原因：frozen_metric_gain；结果指纹：`2287f72b8a1a52166efcaaf863cff51f6c362209b5d21fd167ff8070c8425977`。

问题：when will fairy tail final season be released

初始：March 8, 2018

最终：2018

人工状态：pending，正确性与支持性均未判定。

## 23. NQ / heldout3216 / -1317942101497986450

优先原因：changed_without_frozen_correctness_gain；结果指纹：`b6746d2e55fb31a03ee98aad9ad3a05bdbddd4f24a0b2cbb74672961c238719b`。

问题：when is season 8 for game of thrones

初始：2018 or 2019

最终：July 2016

人工状态：pending，正确性与支持性均未判定。

## 24. NQ / heldout3216 / -1514464247537659362

优先原因：changed_without_frozen_correctness_gain；结果指纹：`3e2d0279719649f91d7b53ddf7c1d6d3830cf947a01a5c35ec2c0c573819c55d`。

问题：when does the sword art online movie take place

初始：after the anime series' second season

最终：after the anime series ' second season , Sword Art Online II

人工状态：pending，正确性与支持性均未判定。

## 25. NQ / heldout3216 / -1910743846708408775

优先原因：changed_without_frozen_correctness_gain；结果指纹：`e60463a01d93059180405bbb40123a2280176a7f02c84a6bacaf1fd71d50a9d0`。

问题：when was the taming ofthe shrew first performed

初始：not specified

最终：between 1590 and 1592

人工状态：pending，正确性与支持性均未判定。

## 26. NQ / heldout3216 / -3276477249293599145

优先原因：changed_without_frozen_correctness_gain；结果指纹：`c69d442b6890a08a5528f2dab06f6802571462eff7d39081319425e3aa601a88`。

问题：when does law of diminishing returns set in

初始：when adding more of a factor of production, while holding all others constant

最终：at some point

人工状态：pending，正确性与支持性均未判定。

## 27. NQ / heldout3216 / -3828702383571778476

优先原因：changed_without_frozen_correctness_gain；结果指纹：`0c8710326635302703e32adb460e56570de9247c05da28023bc87c1c453e9a33`。

问题：what is the meaning of the name von

初始：a noble patrilineality or of/from

最终：a nobiliary particle indicating a noble patrilineality or as a simple preposition

人工状态：pending，正确性与支持性均未判定。

## 28. NQ / heldout3216 / -3984706621862511717

优先原因：changed_without_frozen_correctness_gain；结果指纹：`b8288466ce20f0e9652745ba0f891ae19ba11c093e67987fe999d07466be17e2`。

问题：where does cortisol come from in the body

初始：zona fasciculata of the adrenal cortex

最终：the adrenal gland in the zona fasciculata

人工状态：pending，正确性与支持性均未判定。

## 29. NQ / heldout3216 / -4546405080702371934

优先原因：changed_without_frozen_correctness_gain；结果指纹：`814c6cb842873de901bec72a75855b79c0137fc58cad02d6fbe7b95beb7aec45`。

问题：when was the term prime minister first used

初始：18th century

最终：18th century in the United Kingdom

人工状态：pending，正确性与支持性均未判定。

## 30. NQ / heldout3216 / -4822862932182551070

优先原因：changed_without_frozen_correctness_gain；结果指纹：`5ab91552d217e3b1ab7788e23ef50ed3d839a42c105ad53a39ed250231efeb44`。

问题：another name for a hairpin bend crossword clue

初始：bobby pin

最终：hairpin bend , hairpin corner

人工状态：pending，正确性与支持性均未判定。

## 31. NQ / heldout3216 / -5327690920099764097

优先原因：changed_without_frozen_correctness_gain；结果指纹：`1a12bf4cbc0a6da4c18d5df84a42216cd52f78d9694bbc335b49763e6e764c90`。

问题：where does the name elevator pitch come from

初始：approximately thirty seconds to two minutes

最终：it should be possible to deliver the summary in the time span of an elevator ride

人工状态：pending，正确性与支持性均未判定。

## 32. NQ / heldout3216 / -5551039786517755042

优先原因：changed_without_frozen_correctness_gain；结果指纹：`428225869102bde44a826fa7db967ee7df0d80debd25e91a198a48806507a854`。

问题：who sings there's a hole in my bucket

初始：children

最终：a children 's song

人工状态：pending，正确性与支持性均未判定。

## 33. NQ / heldout3216 / -5623229691925051533

优先原因：changed_without_frozen_correctness_gain；结果指纹：`d002951941e1b278e81c32297ff0051041ce54adc308d7f20bb37119c50b69ca`。

问题：when was the south asian association for regional co-operation (saarc) formed

初始：1985

最终：8 December 1985

人工状态：pending，正确性与支持性均未判定。

## 34. NQ / heldout3216 / -5765931546365345480

优先原因：changed_without_frozen_correctness_gain；结果指纹：`14ec49b0e64110026908d3e197f6239207ed48a4213fcd65f284b722be511a1e`。

问题：where does kevins family go in home alone 2

初始：New York

最终：New York City

人工状态：pending，正确性与支持性均未判定。

## 35. NQ / heldout3216 / -6751220433242447969

优先原因：changed_without_frozen_correctness_gain；结果指纹：`d13c56275dfaf5118f5c0904dfb7fc03b4337f694975c44e7f80456d9c551208`。

问题：how many seasons of the bastard executioner are there

初始：4

最终：one season

人工状态：pending，正确性与支持性均未判定。

## 36. NQ / heldout3216 / -7721509949632484180

优先原因：changed_without_frozen_correctness_gain；结果指纹：`7bf2290699f463a78a340c3fc4c6a63053bcd9d91c42689673489587f97ef34f`。

问题：when was the first driver's license required

初始：August 1, 1910

最终：On August 1 , 1910

人工状态：pending，正确性与支持性均未判定。

## 37. NQ / heldout3216 / -7900537827363346667

优先原因：changed_without_frozen_correctness_gain；结果指纹：`eb6f55fa284baae4a8cfe4672230ae12525fb76b62be832523fc80aca9866fc9`。

问题：where would a subcutaneous injection be made in the skin

初始：under the skin

最终：under the skin , into the fat layer

人工状态：pending，正确性与支持性均未判定。

## 38. NQ / heldout3216 / -8006002028754190814

优先原因：changed_without_frozen_correctness_gain；结果指纹：`44a602d0bd9856f863caf9807583d3a04b30f484424551083d6756954c9634b8`。

问题：who had the most governmental power under the articles of confederation

初始：Congress of the Confederation

最终：the central government 's power was kept quite limited

人工状态：pending，正确性与支持性均未判定。

## 39. NQ / heldout3216 / -8696133314731800801

优先原因：changed_without_frozen_correctness_gain；结果指纹：`81a1ed629bb54e40930334d1411545ca93f4bba42db691164b14db1d3c41ab2c`。

问题：when was the chain first used for f1

初始：Never

最终：1973 Canadian Grand Prix

人工状态：pending，正确性与支持性均未判定。

## 40. NQ / heldout3216 / -8767841388599751671

优先原因：changed_without_frozen_correctness_gain；结果指纹：`ec464cbee5778d2c18a6f5e41b0129a51135b00d5d799a92ed082ca907f90030`。

问题：who coined the phrase let them eat cake

初始：Alphonse Karr

最终：a great princess

人工状态：pending，正确性与支持性均未判定。

## 41. NQ / heldout3216 / 2304123337731564294

优先原因：changed_without_frozen_correctness_gain；结果指纹：`e4ee8282b0456beb880b3c58751508c60880395b79630d25120cb91638375ef4`。

问题：what do the economic systems of the uk germany and russia have in common

初始：None

最终：$2.3 million

人工状态：pending，正确性与支持性均未判定。

## 42. NQ / heldout3216 / 2401915830495655749

优先原因：changed_without_frozen_correctness_gain；结果指纹：`1a930897e3dd0514094ff4b791836a4026ed8b91036d7e64a2f24064ce237552`。

问题：what percentage of the us population lives below the poverty line

初始：15.1%

最终：13.5 %

人工状态：pending，正确性与支持性均未判定。

## 43. NQ / heldout3216 / 2918354452051217435

优先原因：changed_without_frozen_correctness_gain；结果指纹：`ccfdfe74f435741ba579c239913e5f3d537a48a55464001fe84c17c0fc623a3c`。

问题：where was the first high level language operating system created

初始：Germany

最终：Plankalkül , created by Konrad Zuse between 1942 and 1945

人工状态：pending，正确性与支持性均未判定。

## 44. NQ / heldout3216 / 3630480194417653112

优先原因：changed_without_frozen_correctness_gain；结果指纹：`8c37037b5ac2944f10e5a4695df9ca3086e0bb7e2376618e5fc08cc513b1dde5`。

问题：what is henry james the real thing about

初始：the reality-illusion dichotomy

最终：The Real Thing and Other Stories published by Macmillan .

人工状态：pending，正确性与支持性均未判定。

## 45. NQ / heldout3216 / 4140332518759407550

优先原因：changed_without_frozen_correctness_gain；结果指纹：`56948f73f4c92e472c21f7fc1532c26c518684bc522fa814437ab11eae6d6f97`。

问题：where did spock's hand sign come from

初始：Vulcan culture

最终：the Priestly Blessing performed by Jewish Kohanim

人工状态：pending，正确性与支持性均未判定。

## 46. NQ / heldout3216 / 5482109430613356344

优先原因：changed_without_frozen_correctness_gain；结果指纹：`620d8b50c6206a5d89044417ce839f82deb8cb3fdb48a8506ab30cf40d08927b`。

问题：where did the rulers of the qing dynasty originate

初始：Jurchen

最终：a sedentary farming people known as the Jurchen

人工状态：pending，正确性与支持性均未判定。

## 47. NQ / heldout3216 / 6276969298797177537

优先原因：changed_without_frozen_correctness_gain；结果指纹：`cc17b0c4215da066d4393e9492b92b142f61ff4efb959bc8df35ebe5faf0cd52`。

问题：what is the main objective of criminal law

初始：punishment

最终：retribution

人工状态：pending，正确性与支持性均未判定。

## 48. NQ / heldout3216 / 6419695748325791865

优先原因：changed_without_frozen_correctness_gain；结果指纹：`c5e2af902cac4304732d9899be918a30a7996d37bdf4075fb589ad7f816da516`。

问题：who plays david in alvin and the chipmunks

初始：Ross Bagdasarian Jr.

最终：Ross Bagdasarian Sr.

人工状态：pending，正确性与支持性均未判定。

## 49. NQ / heldout3216 / 7206245989456268244

优先原因：changed_without_frozen_correctness_gain；结果指纹：`5f85f6fe56999bcace3551f7928547e88fcc1441f62be452510a4de165190ef0`。

问题：who wrote it's gonna take a lot of love

初始：It's not "It's Gonna Take a Lot of Love", but "It's Gonna Take a Miracle". Written by Teddy Randazzo, Bobby Weinstein, and Lou Stallman.

最终：Teddy Randazzo , Bobby Weinstein , and Lou Stallman

人工状态：pending，正确性与支持性均未判定。

## 50. NQ / heldout3216 / 7520402739582397133

优先原因：changed_without_frozen_correctness_gain；结果指纹：`bbed42c75fe9c8ef7f75d6286e98d018a4dbb709f01e026606e2d5975bffbfd1`。

问题：when was the first book made into a movie

初始：2001

最终：16 November 2001

人工状态：pending，正确性与支持性均未判定。

## 51. NQ / heldout3216 / 795441882608131318

优先原因：changed_without_frozen_correctness_gain；结果指纹：`29fcb889bb16a375858e4f9accb10e9ac89e9b982476250492fe3d174f176544`。

问题：where are the four disney theme parks located

初始：Walt Disney World Resort, Disneyland Paris, Tokyo Disneyland, and none specified for Europe

最终：Lake Buena Vista , Florida

人工状态：pending，正确性与支持性均未判定。

## 52. NQ / heldout3216 / 8803717720023296980

优先原因：changed_without_frozen_correctness_gain；结果指纹：`d2663dfee1f0da8f0d562928a07f399846fcfbe20b86dab3ae784304fb3c62b2`。

问题：where does the sciatic nerve run in the foot

初始：entire lower leg ( except for its inner side )

最终：supplies sensation to the skin of the foot

人工状态：pending，正确性与支持性均未判定。

